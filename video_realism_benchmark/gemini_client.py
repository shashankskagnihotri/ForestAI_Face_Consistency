"""Strict Gemini binary judging with structured yes/no outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
import math
import mimetypes
from pathlib import Path
import time
from typing import Any

from . import config
from .frame_sampling import SampledFrame
from .line_detection import LineDetectionResult, edge_candidate_payload, line_candidate_id
from .prompts import (
    ground_parallel_line_selection_prompt as receding_depth_line_selection_prompt,
    SINGLE_LIGHT_SOURCE_ANALYSIS_PROMPT,
    SINGLE_LIGHT_SOURCE_PROMPT,
    VANISHING_POINT_PROMPT,
    prompt_object_recognizability_prompt,
    prompt_object_visibility_prompt,
)
from .schemas import (
    BINARY_RESPONSE_SCHEMA,
    GEMINI_LINE_SELECTION_RESPONSE_SCHEMA,
    LIGHT_SOURCE_ANALYSIS_RESPONSE_SCHEMA,
    PROMPT_OBJECT_VISIBILITY_RESPONSE_SCHEMA,
    BinaryLabel,
    validate_binary_label,
)
from .utils import display_path, write_json


try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - exercised only on broken envs
    raise ImportError(
        "The official Google GenAI SDK (`google-genai`) is required. Install "
        "dependencies with `pip install -r requirements.txt`."
    ) from exc


@dataclass(frozen=True)
class GeminiEvidence:
    """Evidence files supplied to Gemini for the binary criteria."""

    video_path: Path
    prompt_for_video: str
    sampled_frame_paths: list[Path]
    annotated_frame_paths: list[Path]
    contact_sheet_path: Path
    vanishing_point_diagnostics_path: Path
    runtime_config: config.RuntimeConfig


class GeminiJudgeClient:
    """Small wrapper around the official GenAI SDK with strict validation."""

    def __init__(self, api_key: str, runtime_config: config.RuntimeConfig) -> None:
        if not api_key:
            raise RuntimeError("Gemini API key is empty.")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=config.GEMINI_HTTP_TIMEOUT_MS),
        )
        self.runtime_config = runtime_config
        self._uploaded_video_cache: dict[Path, Any] = {}

    def _mime_type(self, path: Path) -> str:
        """Infer a MIME type and fail if the file type is unknown."""

        guessed, _ = mimetypes.guess_type(path.as_posix())
        if guessed is None:
            raise RuntimeError(f"Could not infer MIME type for Gemini evidence file: {path}")
        return guessed

    def _part_from_file_bytes(self, path: Path) -> Any:
        """Attach a local image/JSON evidence file as bytes."""

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Gemini evidence file does not exist: {path}")
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=self._mime_type(path))

    def _file_state_name(self, file_obj: Any) -> str | None:
        """Extract a stable file-state name from the SDK object."""

        state = getattr(file_obj, "state", None)
        if state is None:
            return None
        return str(getattr(state, "name", state)).upper()

    def _upload_video_once(self, video_path: Path) -> Any:
        """Upload the full video once and wait until Gemini can consume it."""

        resolved = video_path.resolve()
        if resolved in self._uploaded_video_cache:
            return self._uploaded_video_cache[resolved]
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Video for Gemini upload does not exist: {resolved}")

        try:
            uploaded = self.client.files.upload(file=str(resolved))
        except Exception as exc:
            raise RuntimeError(f"Failed to upload video to Gemini: {resolved}") from exc

        name = getattr(uploaded, "name", None)
        if not name:
            raise RuntimeError("Gemini file upload did not return a file name.")

        deadline = time.time() + config.GEMINI_FILE_PROCESSING_TIMEOUT_SEC
        latest = uploaded
        while True:
            state_name = self._file_state_name(latest)
            if state_name is None or state_name == "ACTIVE":
                self._uploaded_video_cache[resolved] = latest
                return latest
            if state_name == "FAILED":
                raise RuntimeError(f"Gemini video processing failed for uploaded file: {name}")
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for Gemini video processing: {name}")
            time.sleep(config.GEMINI_FILE_POLL_INTERVAL_SEC)
            latest = self.client.files.get(name=name)

    def _serialize_response(self, response: Any) -> dict[str, Any]:
        """Persist generated text plus a compact SDK dump for failed attempts."""

        payload: dict[str, Any] = {"text": getattr(response, "text", None)}
        if hasattr(response, "model_dump"):
            try:
                payload["response"] = response.model_dump(mode="json", exclude_none=True)
            except Exception:
                payload["response_repr"] = repr(response)
        return payload

    def _generate_content_with_retries(
        self,
        criterion_name: str,
        contents: list[Any],
        generation_config: Any,
    ) -> Any:
        """Call Gemini with bounded request time and a small retry budget."""

        last_exc: Exception | None = None
        for attempt in range(1, config.GEMINI_REQUEST_MAX_ATTEMPTS + 1):
            try:
                return self.client.models.generate_content(
                    model=config.GEMINI_MODEL_NAME,
                    contents=contents,
                    config=generation_config,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= config.GEMINI_REQUEST_MAX_ATTEMPTS:
                    break
                time.sleep(config.GEMINI_REQUEST_RETRY_SLEEP_SEC)
        raise RuntimeError(
            f"Gemini request failed for criterion '{criterion_name}' after "
            f"{config.GEMINI_REQUEST_MAX_ATTEMPTS} attempt(s) using required "
            f"model '{config.GEMINI_MODEL_NAME}'."
        ) from last_exc

    def _generate_binary(
        self,
        criterion_name: str,
        prompt: str,
        contents: list[Any],
        request_metadata: dict[str, Any],
    ) -> BinaryLabel:
        """Call Gemini with enum-structured output and validate yes/no exactly."""

        request_path = self.runtime_config.gemini_requests_dir / f"{criterion_name}_request.json"
        response_path = self.runtime_config.gemini_responses_dir / f"{criterion_name}_response.json"
        full_request_metadata = {
            "criterion": criterion_name,
            "model": config.GEMINI_MODEL_NAME,
            "http_timeout_ms": config.GEMINI_HTTP_TIMEOUT_MS,
            "max_attempts": config.GEMINI_REQUEST_MAX_ATTEMPTS,
            "response_mime_type": "text/x.enum",
            "response_schema": BINARY_RESPONSE_SCHEMA,
            "prompt": prompt,
            **request_metadata,
        }
        write_json(request_path, full_request_metadata)

        generation_config = types.GenerateContentConfig(
            response_mime_type="text/x.enum",
            response_schema=BINARY_RESPONSE_SCHEMA,
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
            max_output_tokens=config.GEMINI_BINARY_MAX_OUTPUT_TOKENS,
        )

        last_error: Exception | None = None
        for parse_attempt in range(1, config.GEMINI_JSON_PARSE_MAX_ATTEMPTS + 1):
            response = self._generate_content_with_retries(criterion_name, contents, generation_config)
            response_payload = self._serialize_response(response)
            try:
                decision = validate_binary_label(response_payload.get("text"))
            except ValueError as exc:
                last_error = exc
                invalid_path = (
                    self.runtime_config.gemini_responses_dir
                    / f"{criterion_name}_invalid_attempt_{parse_attempt:02d}.json"
                )
                write_json(
                    invalid_path,
                    {
                        "error": str(exc),
                        "serialized_response": response_payload,
                    },
                )
                if parse_attempt < config.GEMINI_JSON_PARSE_MAX_ATTEMPTS:
                    time.sleep(config.GEMINI_REQUEST_RETRY_SLEEP_SEC)
                continue
            write_json(response_path, {criterion_name: decision})
            return decision

        raise RuntimeError(
            f"Gemini did not return a valid binary label for {criterion_name} after "
            f"{config.GEMINI_JSON_PARSE_MAX_ATTEMPTS} parse attempt(s)."
        ) from last_error

    def _generate_json(
        self,
        criterion_name: str,
        prompt: str,
        contents: list[Any],
        request_metadata: dict[str, Any],
        response_schema: dict[str, object],
    ) -> dict[str, Any]:
        """Call Gemini with JSON-structured output and parse the response text."""

        request_path = self.runtime_config.gemini_requests_dir / f"{criterion_name}_request.json"
        response_path = self.runtime_config.gemini_responses_dir / f"{criterion_name}_response.json"
        full_request_metadata = {
            "criterion": criterion_name,
            "model": config.GEMINI_MODEL_NAME,
            "http_timeout_ms": config.GEMINI_HTTP_TIMEOUT_MS,
            "max_attempts": config.GEMINI_REQUEST_MAX_ATTEMPTS,
            "json_parse_max_attempts": config.GEMINI_JSON_PARSE_MAX_ATTEMPTS,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "prompt": prompt,
            **request_metadata,
        }
        write_json(request_path, full_request_metadata)

        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
            max_output_tokens=config.GEMINI_JSON_MAX_OUTPUT_TOKENS,
        )

        last_error: Exception | None = None
        for parse_attempt in range(1, config.GEMINI_JSON_PARSE_MAX_ATTEMPTS + 1):
            response = self._generate_content_with_retries(criterion_name, contents, generation_config)
            response_payload = self._serialize_response(response)
            response_text = response_payload.get("text")
            invalid_path = (
                self.runtime_config.gemini_responses_dir
                / f"{criterion_name}_invalid_attempt_{parse_attempt:02d}.json"
            )
            if not isinstance(response_text, str):
                last_error = RuntimeError(
                    f"Gemini returned non-text JSON response for {criterion_name}: {response_text!r}"
                )
                write_json(
                    invalid_path,
                    {
                        "error": str(last_error),
                        "serialized_response": response_payload,
                    },
                )
            else:
                try:
                    parsed = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    last_error = RuntimeError(
                        f"Gemini returned invalid JSON for {criterion_name}: {response_text!r}"
                    )
                    write_json(
                        invalid_path,
                        {
                            "error": str(exc),
                            "raw_text": response_text,
                        },
                    )
                else:
                    if not isinstance(parsed, dict):
                        last_error = RuntimeError(
                            f"Gemini JSON response for {criterion_name} was not an object: {parsed!r}"
                        )
                        write_json(
                            invalid_path,
                            {
                                "error": str(last_error),
                                "parsed_response": parsed,
                            },
                        )
                    else:
                        write_json(response_path, parsed)
                        return parsed
            if parse_attempt < config.GEMINI_JSON_PARSE_MAX_ATTEMPTS:
                time.sleep(config.GEMINI_REQUEST_RETRY_SLEEP_SEC)

        raise RuntimeError(
            f"Gemini did not return valid JSON for {criterion_name} after "
            f"{config.GEMINI_JSON_PARSE_MAX_ATTEMPTS} parse attempt(s)."
        ) from last_error


def _bounded_float(value: object, low: float, high: float) -> float:
    """Convert a numeric Gemini value and clamp it to image bounds."""

    numeric = float(value)
    if numeric < low:
        return low
    if numeric > high:
        return high
    return numeric


def _line_orientation_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return line orientation modulo 180 degrees."""

    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _orientation_delta_deg(a: float, b: float) -> float:
    """Smallest absolute orientation difference modulo 180 degrees."""

    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def _line_outside_central_occluder_fraction(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> float:
    """Estimate how much of a line lies outside the central person/map region."""

    outside = 0
    sample_count = config.LINE_CONFIDENCE_SAMPLE_COUNT
    x_min = config.LINE_CENTER_OCCLUDER_X_MIN_FRACTION * width
    x_max = config.LINE_CENTER_OCCLUDER_X_MAX_FRACTION * width
    y_min = config.LINE_CENTER_OCCLUDER_Y_MIN_FRACTION * height
    y_max = config.LINE_CENTER_OCCLUDER_Y_MAX_FRACTION * height
    for index in range(sample_count):
        t = index / float(sample_count - 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            outside += 1
    return outside / float(sample_count)


def _sanitize_line_selection_payload(
    payload: dict[str, Any],
    sampled_frame: SampledFrame,
    project_root: Path,
) -> dict[str, Any]:
    """Validate and normalize Gemini-selected finite edge segments."""

    raw_lines = payload.get("lines", [])
    if not isinstance(raw_lines, list):
        raw_lines = []

    sanitized_lines: list[dict[str, object]] = []
    for raw_line in raw_lines[: config.GEMINI_LINE_SELECTION_MAX_LINES_PER_FRAME]:
        if not isinstance(raw_line, dict):
            continue
        try:
            x1 = _bounded_float(raw_line.get("x1"), 0.0, float(sampled_frame.width - 1))
            y1 = _bounded_float(raw_line.get("y1"), 0.0, float(sampled_frame.height - 1))
            x2 = _bounded_float(raw_line.get("x2"), 0.0, float(sampled_frame.width - 1))
            y2 = _bounded_float(raw_line.get("y2"), 0.0, float(sampled_frame.height - 1))
            confidence = _bounded_float(raw_line.get("confidence", 0.0), 0.0, 1.0)
        except (TypeError, ValueError):
            continue

        length = ((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0) ** 0.5
        if length < config.GEMINI_LINE_SELECTION_MIN_LENGTH_PX:
            continue
        if confidence < config.GEMINI_LINE_SELECTION_MIN_CONFIDENCE:
            continue
        if abs(y2 - y1) / length < config.GEMINI_LINE_SELECTION_MIN_VERTICAL_CHANGE_FRACTION:
            continue
        angle = _line_orientation_deg(x1, y1, x2, y2)
        if _orientation_delta_deg(angle, 90.0) < config.GEMINI_LINE_SELECTION_VERTICAL_REJECTION_ANGLE_DEG:
            continue
        outside_central_occluder_fraction = _line_outside_central_occluder_fraction(
            x1,
            y1,
            x2,
            y2,
            sampled_frame.width,
            sampled_frame.height,
        )
        if outside_central_occluder_fraction < config.GEMINI_LINE_SELECTION_CENTER_OCCLUDER_MIN_OUTSIDE_FRACTION:
            continue
        margin = config.GEMINI_LINE_SELECTION_BORDER_MARGIN_PX
        both_on_top_or_bottom_border = (
            (y1 <= margin and y2 <= margin)
            or (y1 >= sampled_frame.height - 1 - margin and y2 >= sampled_frame.height - 1 - margin)
        )
        both_on_left_or_right_border = (
            (x1 <= margin and x2 <= margin)
            or (x1 >= sampled_frame.width - 1 - margin and x2 >= sampled_frame.width - 1 - margin)
        )
        if both_on_top_or_bottom_border or both_on_left_or_right_border:
            continue

        rationale = str(
            raw_line.get(
                "why_this_projects_away_from_camera",
                raw_line.get("why_this_is_ground_parallel", ""),
            )
        )
        sanitized_lines.append(
            {
                "line_id": str(raw_line.get("line_id", f"line_{len(sanitized_lines) + 1}")),
                "object_or_scene_edge": str(raw_line.get("object_or_scene_edge", ""))[:200],
                "why_this_projects_away_from_camera": rationale[:300],
                "confidence": round(confidence, 6),
                "x1": round(x1, 3),
                "y1": round(y1, 3),
                "x2": round(x2, 3),
                "y2": round(y2, 3),
                "length_px": round(length, 3),
                "angle_deg": round(angle, 3),
                "outside_central_occluder_fraction": round(outside_central_occluder_fraction, 6),
            }
        )

    usable = bool(payload.get("usable")) and len(sanitized_lines) >= 2
    return {
        "frame_index": sampled_frame.frame_index,
        "timestamp_sec": round(sampled_frame.timestamp_sec, 6),
        "image_width": sampled_frame.width,
        "image_height": sampled_frame.height,
        "usable": usable,
        "frame_assessment": str(payload.get("frame_assessment", ""))[:500],
        "lines": sanitized_lines if usable else [],
        "raw_sampled_frame_path": display_path(sampled_frame.image_path, project_root),
    }


def _sanitize_edge_candidate_selection_payload(
    payload: dict[str, Any],
    sampled_frame: SampledFrame,
    candidate_result: LineDetectionResult,
    project_root: Path,
    candidate_overlay_path: Path,
    candidate_manifest_path: Path,
) -> dict[str, Any]:
    """Validate Gemini-selected candidate IDs and map them to real edge lines."""

    raw_ids = payload.get("selected_candidate_ids", [])
    if not isinstance(raw_ids, list):
        raw_ids = []
    candidate_by_id = {
        line_candidate_id(segment): segment
        for segment in candidate_result.accepted_segments
    }

    selected_ids: list[str] = []
    for raw_id in raw_ids:
        candidate_id = str(raw_id)
        if candidate_id not in candidate_by_id:
            continue
        if candidate_id in selected_ids:
            continue
        selected_ids.append(candidate_id)
        if len(selected_ids) >= config.GEMINI_LINE_SELECTION_MAX_LINES_PER_FRAME:
            break

    selected_lines: list[dict[str, object]] = []
    for candidate_id in selected_ids:
        segment = candidate_by_id[candidate_id]
        line_payload = edge_candidate_payload(segment)
        line_payload.update(
            {
                "line_id": candidate_id,
                "object_or_scene_edge": "edge-tensor-derived scene edge candidate",
                "why_this_projects_away_from_camera": (
                    "Gemini selected this real edge candidate ID as receding-depth evidence."
                ),
                "confidence": round(float(segment.edge_confidence), 6),
                "geometry_source": "full_resolution_edge_probability_tensor",
            }
        )
        selected_lines.append(line_payload)

    usable = bool(payload.get("usable")) and len(selected_lines) >= 2
    return {
        "frame_index": sampled_frame.frame_index,
        "timestamp_sec": round(sampled_frame.timestamp_sec, 6),
        "image_width": sampled_frame.width,
        "image_height": sampled_frame.height,
        "usable": usable,
        "frame_assessment": str(payload.get("frame_assessment", ""))[:500],
        "selected_candidate_ids": selected_ids if usable else [],
        "lines": selected_lines if usable else [],
        "edge_candidate_count": len(candidate_result.accepted_segments),
        "edge_candidate_overlay_path": display_path(candidate_overlay_path, project_root),
        "edge_candidate_manifest_path": display_path(candidate_manifest_path, project_root),
        "raw_sampled_frame_path": display_path(sampled_frame.image_path, project_root),
    }


def select_receding_depth_lines(
    client: GeminiJudgeClient,
    sampled_frame: SampledFrame,
    prompt_for_video: str,
    candidate_result: LineDetectionResult,
    candidate_overlay_path: Path,
) -> dict[str, Any]:
    """Ask Gemini to select real edge-tensor candidate IDs for VP evidence."""

    prompt = receding_depth_line_selection_prompt(
        frame_index=sampled_frame.frame_index,
        timestamp_sec=sampled_frame.timestamp_sec,
        width=sampled_frame.width,
        height=sampled_frame.height,
        max_lines=config.GEMINI_LINE_SELECTION_MAX_LINES_PER_FRAME,
        prompt_for_video=prompt_for_video,
    )
    criterion_name = f"receding_depth_lines_frame_{sampled_frame.frame_index:06d}"
    candidate_manifest = {
        "frame_index": sampled_frame.frame_index,
        "geometry_source": "full_resolution_edge_probability_tensor",
        "instruction": (
            "Select only candidate_id values from this list. Do not invent coordinates."
        ),
        "candidates": [
            edge_candidate_payload(segment)
            for segment in candidate_result.accepted_segments
        ],
    }
    candidate_manifest_path = (
        client.runtime_config.gemini_requests_dir
        / f"{criterion_name}_edge_candidates.json"
    )
    write_json(candidate_manifest_path, candidate_manifest)
    metadata = {
        "evidence_files": {
            "sampled_frame": display_path(sampled_frame.image_path, client.runtime_config.project_root),
            "edge_candidate_overlay": display_path(candidate_overlay_path, client.runtime_config.project_root),
            "edge_candidate_manifest": display_path(candidate_manifest_path, client.runtime_config.project_root),
        },
        "video_generation_prompt": prompt_for_video,
        "frame_index": sampled_frame.frame_index,
        "timestamp_sec": round(sampled_frame.timestamp_sec, 6),
        "image_width": sampled_frame.width,
        "image_height": sampled_frame.height,
        "edge_candidate_count": len(candidate_result.accepted_segments),
    }
    payload = client._generate_json(
        criterion_name=criterion_name,
        prompt=prompt,
        contents=[
            prompt,
            client._part_from_file_bytes(sampled_frame.image_path),
            client._part_from_file_bytes(candidate_overlay_path),
            client._part_from_file_bytes(candidate_manifest_path),
        ],
        request_metadata=metadata,
        response_schema=GEMINI_LINE_SELECTION_RESPONSE_SCHEMA,
    )
    sanitized = _sanitize_edge_candidate_selection_payload(
        payload,
        sampled_frame,
        candidate_result,
        client.runtime_config.project_root,
        candidate_overlay_path,
        candidate_manifest_path,
    )
    response_path = client.runtime_config.gemini_responses_dir / f"{criterion_name}_response.json"
    if response_path.exists():
        response_payload = json.loads(response_path.read_text(encoding="utf-8"))
        response_payload["sanitized_selection"] = sanitized
        write_json(response_path, response_payload)
    return sanitized


select_ground_parallel_lines = select_receding_depth_lines


def _raw_frame_parts(client: GeminiJudgeClient, evidence: GeminiEvidence) -> list[Any]:
    """Create byte parts for sampled raw frames."""

    return [client._part_from_file_bytes(path) for path in evidence.sampled_frame_paths]


def _frame_manifest(evidence: GeminiEvidence) -> list[dict[str, object]]:
    """Return frame labels that Gemini can use in compact per-frame JSON."""

    manifest: list[dict[str, object]] = []
    for path in evidence.sampled_frame_paths:
        frame_index = None
        timestamp_sec = None
        stem_parts = path.stem.split("_")
        if len(stem_parts) >= 4 and stem_parts[0] == "frame":
            try:
                frame_index = int(stem_parts[1])
                timestamp_sec = float(stem_parts[3].removesuffix("s"))
            except ValueError:
                frame_index = None
                timestamp_sec = None
        manifest.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "path": display_path(path, evidence.runtime_config.project_root),
            }
        )
    return manifest


def _labeled_raw_frame_parts(client: GeminiJudgeClient, evidence: GeminiEvidence) -> list[Any]:
    """Interleave labels and sampled-frame image parts for frame-level analysis."""

    parts: list[Any] = []
    for frame, path in zip(_frame_manifest(evidence), evidence.sampled_frame_paths, strict=True):
        parts.append(
            "Sampled frame "
            f"frame_index={frame['frame_index']} "
            f"timestamp_sec={frame['timestamp_sec']}"
        )
        parts.append(client._part_from_file_bytes(path))
    return parts


def _normalize_light_source_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only the compact fields requested for light-source analysis."""

    decision = validate_binary_label(payload.get("single_light_source_consistency"))
    raw_sources = payload.get("light_sources", [])
    if not isinstance(raw_sources, list):
        raw_sources = []

    light_sources: list[dict[str, str]] = []
    for index, raw_source in enumerate(raw_sources, start=1):
        if not isinstance(raw_source, dict):
            continue
        direction = str(raw_source.get("relative_direction_from_frame_center", "unknown"))
        if direction not in config.LIGHT_SOURCE_DIRECTIONS:
            direction = "unknown"
        source_id = str(raw_source.get("source_id") or f"light_{index}")[:80]
        light_sources.append(
            {
                "source_id": source_id,
                "relative_direction_from_frame_center": direction,
            }
        )

    return {
        "single_light_source_consistency": decision,
        "light_sources": light_sources,
    }


def _normalize_prompt_object_visibility(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only object-level and frame-level visibility fields."""

    decision = validate_binary_label(payload.get("prompt_object_recognizability"))
    raw_objects = payload.get("objects_of_interest", [])
    if not isinstance(raw_objects, list):
        raw_objects = []

    objects: list[dict[str, object]] = []
    for raw_object in raw_objects:
        if not isinstance(raw_object, dict):
            continue
        raw_frames = raw_object.get("per_frame_visibility", [])
        if not isinstance(raw_frames, list):
            raw_frames = []
        per_frame_visibility: list[dict[str, object]] = []
        for raw_frame in raw_frames:
            if not isinstance(raw_frame, dict):
                continue
            try:
                frame_index = int(float(raw_frame.get("frame_index")))
                timestamp_sec = round(float(raw_frame.get("timestamp_sec")), 6)
            except (TypeError, ValueError):
                continue
            per_frame_visibility.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "visible": bool(raw_frame.get("visible")),
                }
            )
        object_name = str(raw_object.get("object_name") or "object_of_interest")[:120]
        objects.append(
            {
                "object_name": object_name,
                "visible_overall": bool(raw_object.get("visible_overall")),
                "per_frame_visibility": per_frame_visibility,
            }
        )

    return {
        "prompt_object_recognizability": decision,
        "objects_of_interest": objects,
    }


def _write_compact_vanishing_point_diagnostics(evidence: GeminiEvidence) -> Path:
    """Write compact VP diagnostics for Gemini without thousands of rejected segments."""

    full_diagnostics = json.loads(evidence.vanishing_point_diagnostics_path.read_text(encoding="utf-8"))
    compact_diagnostics: list[dict[str, Any]] = []
    segment_keys = (
        "segment_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "length",
        "angle_deg",
        "midpoint_x",
        "midpoint_y",
        "edge_confidence",
        "edge_support_fraction",
        "family_label",
        "vp_inlier",
        "vp_residual_deg",
        "object_or_scene_edge",
        "selection_rationale",
        "selection_confidence",
    )

    for frame in full_diagnostics:
        rejected_summary: dict[str, int] = {}
        for segment in frame.get("rejected_line_segments", []):
            reason = segment.get("rejection_reason") or "unknown"
            rejected_summary[reason] = rejected_summary.get(reason, 0) + 1

        compact_diagnostics.append(
            {
                "frame_index": frame.get("frame_index"),
                "timestamp_sec": frame.get("timestamp_sec"),
                "number_detected_line_segments": frame.get("number_detected_line_segments"),
                "number_candidate_structural_line_segments": frame.get(
                    "number_candidate_structural_line_segments"
                ),
                "number_vanishing_point_inliers": frame.get("number_vanishing_point_inliers"),
                "inlier_ratio": frame.get("inlier_ratio"),
                "residual_mean": frame.get("residual_mean"),
                "residual_median": frame.get("residual_median"),
                "residual_max": frame.get("residual_max"),
                "estimated_vanishing_point_coordinates": frame.get(
                    "estimated_vanishing_point_coordinates"
                ),
                "vanishing_point_inside_image": frame.get("vanishing_point_inside_image"),
                "geometrically_consistent_with_one_dominant_vanishing_point": frame.get(
                    "geometrically_consistent_with_one_dominant_vanishing_point"
                ),
                "raw_sampled_frame_path": frame.get("raw_sampled_frame_path"),
                "annotated_frame_path": frame.get("annotated_frame_path"),
                "rejected_line_segment_summary": rejected_summary,
                "accepted_line_segments": [
                    {key: segment.get(key) for key in segment_keys if key in segment}
                    for segment in frame.get("accepted_line_segments", [])
                ],
            }
        )

    compact_path = evidence.runtime_config.result_dir / "vanishing_point_compact_diagnostics.json"
    write_json(compact_path, compact_diagnostics)
    return compact_path


def evaluate_vanishing_point(client: GeminiJudgeClient, evidence: GeminiEvidence) -> BinaryLabel:
    """Return exactly yes/no for vanishing-point consistency."""

    compact_diagnostics_path = _write_compact_vanishing_point_diagnostics(evidence)
    contents: list[Any] = [VANISHING_POINT_PROMPT]
    contents.append(client._part_from_file_bytes(evidence.contact_sheet_path))
    contents.append(client._part_from_file_bytes(compact_diagnostics_path))
    metadata = {
        "evidence_files": {
            "contact_sheet": display_path(evidence.contact_sheet_path, evidence.runtime_config.project_root),
            "full_vanishing_point_diagnostics_saved_on_disk": display_path(
                evidence.vanishing_point_diagnostics_path,
                evidence.runtime_config.project_root,
            ),
            "compact_vanishing_point_diagnostics_sent_to_gemini": display_path(
                compact_diagnostics_path,
                evidence.runtime_config.project_root,
            ),
        }
    }
    return client._generate_binary("vanishing_point", VANISHING_POINT_PROMPT, contents, metadata)


def evaluate_single_light_source(client: GeminiJudgeClient, evidence: GeminiEvidence) -> BinaryLabel:
    """Return yes/no while saving compact light-source direction JSON."""

    uploaded_video = client._upload_video_once(evidence.video_path)
    contents: list[Any] = [
        SINGLE_LIGHT_SOURCE_ANALYSIS_PROMPT,
        uploaded_video,
        f"Sampled frame manifest: {json.dumps(_frame_manifest(evidence), ensure_ascii=True)}",
    ]
    contents.extend(_labeled_raw_frame_parts(client, evidence))
    metadata = {
        "evidence_files": {
            "video": display_path(evidence.video_path, evidence.runtime_config.project_root),
            "sampled_frames": [
                display_path(path, evidence.runtime_config.project_root)
                for path in evidence.sampled_frame_paths
            ],
        },
        "frame_manifest": _frame_manifest(evidence),
        "uploaded_video_name": getattr(uploaded_video, "name", None),
        "uploaded_video_uri": getattr(uploaded_video, "uri", None),
    }
    payload = client._generate_json(
        criterion_name="single_light_source",
        prompt=SINGLE_LIGHT_SOURCE_ANALYSIS_PROMPT,
        contents=contents,
        request_metadata=metadata,
        response_schema=LIGHT_SOURCE_ANALYSIS_RESPONSE_SCHEMA,
    )
    normalized = _normalize_light_source_analysis(payload)
    write_json(client.runtime_config.gemini_responses_dir / "single_light_source_response.json", normalized)
    return validate_binary_label(normalized["single_light_source_consistency"])


def evaluate_prompt_object_recognizability(client: GeminiJudgeClient, evidence: GeminiEvidence) -> BinaryLabel:
    """Return yes/no while saving compact prompted-object visibility JSON."""

    prompt = prompt_object_visibility_prompt(evidence.prompt_for_video)
    uploaded_video = client._upload_video_once(evidence.video_path)
    contents: list[Any] = [
        prompt,
        uploaded_video,
        f"Sampled frame manifest: {json.dumps(_frame_manifest(evidence), ensure_ascii=True)}",
    ]
    contents.extend(_labeled_raw_frame_parts(client, evidence))
    metadata = {
        "evidence_files": {
            "video": display_path(evidence.video_path, evidence.runtime_config.project_root),
            "sampled_frames": [
                display_path(path, evidence.runtime_config.project_root)
                for path in evidence.sampled_frame_paths
            ],
        },
        "frame_manifest": _frame_manifest(evidence),
        "uploaded_video_name": getattr(uploaded_video, "name", None),
        "uploaded_video_uri": getattr(uploaded_video, "uri", None),
    }
    payload = client._generate_json(
        criterion_name="prompt_object_recognizability",
        prompt=prompt,
        contents=contents,
        request_metadata=metadata,
        response_schema=PROMPT_OBJECT_VISIBILITY_RESPONSE_SCHEMA,
    )
    normalized = _normalize_prompt_object_visibility(payload)
    write_json(
        client.runtime_config.gemini_responses_dir / "prompt_object_recognizability_response.json",
        normalized,
    )
    return validate_binary_label(normalized["prompt_object_recognizability"])
