"""Line-segment extraction and deterministic structural filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable

import cv2
import numpy as np

from . import config


@dataclass
class LineSegment:
    """A detected line segment with filtering and VP-inlier metadata."""

    segment_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    midpoint_x: float
    midpoint_y: float
    edge_confidence: float
    edge_support_fraction: float
    accepted: bool
    rejection_reason: str | None
    family_label: str | None
    vp_inlier: bool = False
    vp_residual_deg: float | None = None
    object_or_scene_edge: str | None = None
    selection_rationale: str | None = None
    selection_confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the segment for diagnostics."""

        return asdict(self)


@dataclass(frozen=True)
class LineDetectionResult:
    """Accepted and rejected lines for one frame."""

    frame_index: int
    accepted_segments: list[LineSegment]
    rejected_segments: list[LineSegment]

    @property
    def all_segments(self) -> list[LineSegment]:
        return self.accepted_segments + self.rejected_segments


def _angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return orientation modulo 180 degrees."""

    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle % 180.0


def _orientation_delta_deg(a: float, b: float) -> float:
    """Smallest absolute orientation difference modulo 180 degrees."""

    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def _edge_statistics(edge_probability_map: np.ndarray, line: tuple[float, float, float, float]) -> tuple[float, float]:
    """Mean edge probability and support fraction sampled along a segment."""

    x1, y1, x2, y2 = line
    xs = np.linspace(x1, x2, config.LINE_CONFIDENCE_SAMPLE_COUNT)
    ys = np.linspace(y1, y2, config.LINE_CONFIDENCE_SAMPLE_COUNT)
    xs_i = np.clip(np.round(xs).astype(np.int32), 0, edge_probability_map.shape[1] - 1)
    ys_i = np.clip(np.round(ys).astype(np.int32), 0, edge_probability_map.shape[0] - 1)
    samples = edge_probability_map[ys_i, xs_i]
    mean_probability = float(np.mean(samples))
    support_fraction = float(np.mean(samples >= config.EDGE_PROBABILITY_THRESHOLD))
    return mean_probability, support_fraction


def _center_occluder_outside_fraction(segment: LineSegment, width: int, height: int) -> float:
    """Estimate how much of a segment lies outside the central person/map region."""

    xs = np.linspace(segment.x1, segment.x2, config.LINE_CONFIDENCE_SAMPLE_COUNT)
    ys = np.linspace(segment.y1, segment.y2, config.LINE_CONFIDENCE_SAMPLE_COUNT)
    x_min = config.LINE_CENTER_OCCLUDER_X_MIN_FRACTION * width
    x_max = config.LINE_CENTER_OCCLUDER_X_MAX_FRACTION * width
    y_min = config.LINE_CENTER_OCCLUDER_Y_MIN_FRACTION * height
    y_max = config.LINE_CENTER_OCCLUDER_Y_MAX_FRACTION * height
    inside = (x_min <= xs) & (xs <= x_max) & (y_min <= ys) & (ys <= y_max)
    return float(np.mean(~inside))


def _is_frame_border_line(segment: LineSegment, width: int, height: int) -> bool:
    """Reject the image border itself as a non-scene structural line."""

    margin = config.LINE_BORDER_MARGIN_FRACTION * float(max(width, height))
    near_left = abs(segment.x1) <= margin and abs(segment.x2) <= margin
    near_right = abs(segment.x1 - (width - 1)) <= margin and abs(segment.x2 - (width - 1)) <= margin
    near_top = abs(segment.y1) <= margin and abs(segment.y2) <= margin
    near_bottom = abs(segment.y1 - (height - 1)) <= margin and abs(segment.y2 - (height - 1)) <= margin
    return near_left or near_right or near_top or near_bottom


def _is_near_duplicate(candidate: LineSegment, kept: Iterable[LineSegment]) -> bool:
    """Suppress near-duplicate LSD outputs while retaining the strongest line."""

    c_points = np.array([[candidate.x1, candidate.y1], [candidate.x2, candidate.y2]], dtype=np.float32)
    for segment in kept:
        if _orientation_delta_deg(candidate.angle_deg, segment.angle_deg) > config.LINE_DUPLICATE_ANGLE_DEG:
            continue
        midpoint_distance = math.hypot(
            candidate.midpoint_x - segment.midpoint_x,
            candidate.midpoint_y - segment.midpoint_y,
        )
        if midpoint_distance > config.LINE_DUPLICATE_MIDPOINT_PX:
            continue
        s_points = np.array([[segment.x1, segment.y1], [segment.x2, segment.y2]], dtype=np.float32)
        direct = float(np.mean(np.linalg.norm(c_points - s_points, axis=1)))
        flipped = float(np.mean(np.linalg.norm(c_points - s_points[::-1], axis=1)))
        if min(direct, flipped) <= config.LINE_DUPLICATE_ENDPOINT_PX:
            return True
    return False


def _family_label(angle_deg: float) -> str:
    """Assign a deterministic orientation-bin label."""

    bin_index = int(angle_deg // config.LINE_ORIENTATION_BIN_DEG)
    bin_start = bin_index * config.LINE_ORIENTATION_BIN_DEG
    return f"orientation_{bin_start:.0f}_{bin_start + config.LINE_ORIENTATION_BIN_DEG:.0f}"


def _select_orientation_families(candidates: list[LineSegment], image_diagonal: float) -> set[str]:
    """Keep line families with enough repeated straight structural support."""

    support: dict[str, float] = {}
    counts: dict[str, int] = {}
    for segment in candidates:
        label = _family_label(segment.angle_deg)
        support[label] = support.get(label, 0.0) + segment.length
        counts[label] = counts.get(label, 0) + 1

    minimum_support = config.LINE_MIN_ORIENTATION_FAMILY_SUPPORT_FRACTION * image_diagonal
    ranked = sorted(support, key=lambda label: (-support[label], label))
    selected: set[str] = set()
    for label in ranked:
        if counts[label] < config.LINE_MIN_ORIENTATION_FAMILY_COUNT:
            continue
        if support[label] < minimum_support:
            continue
        selected.add(label)
        if len(selected) >= config.LINE_MAX_STRUCTURAL_FAMILIES:
            break
    return selected


def _is_ground_plane_parallel_candidate(
    segment: LineSegment,
    height: int,
    minimum_length: float,
) -> bool:
    """Keep only lower-scene straight candidates likely to be receding depth cues.

    This heuristic intentionally rejects upper-frame silhouettes and decorative
    tangents before visualization, so overlaid evidence is limited to plausible
    floor/corridor perspective lines rather than every strong edge.
    """

    midpoint_y_fraction = segment.midpoint_y / float(height)
    lower_endpoint_fraction = max(segment.y1, segment.y2) / float(height)
    if midpoint_y_fraction >= config.LINE_GROUND_PLANE_MIN_MIDPOINT_Y_FRACTION:
        return True
    return (
        lower_endpoint_fraction >= config.LINE_GROUND_PLANE_MIN_LOWER_ENDPOINT_Y_FRACTION
        and segment.length >= config.LINE_GROUND_PLANE_LONG_LINE_MULTIPLIER * minimum_length
    )


def _ground_plane_priority(segment: LineSegment, width: int, height: int) -> float:
    """Rank plausible lower-scene structural lines ahead of texture/tangent clutter."""

    midpoint_y_fraction = segment.midpoint_y / float(height)
    lower_endpoint_fraction = max(segment.y1, segment.y2) / float(height)
    lower_scene_weight = 0.5 + min(1.0, max(midpoint_y_fraction, lower_endpoint_fraction))
    center_occluder_weight = 0.25 + 0.75 * _center_occluder_outside_fraction(segment, width, height)
    return segment.length * max(segment.edge_confidence, 0.01) * lower_scene_weight * center_occluder_weight


def line_candidate_id(segment: LineSegment) -> str:
    """Return the stable human-facing ID for an edge-derived candidate line."""

    return f"E{segment.segment_id:02d}"


def edge_candidate_payload(segment: LineSegment) -> dict[str, object]:
    """Return compact JSON for a real edge-tensor-derived candidate line."""

    return {
        "candidate_id": line_candidate_id(segment),
        "x1": round(segment.x1, 3),
        "y1": round(segment.y1, 3),
        "x2": round(segment.x2, 3),
        "y2": round(segment.y2, 3),
        "length_px": round(segment.length, 3),
        "angle_deg": round(segment.angle_deg, 3),
        "midpoint_x": round(segment.midpoint_x, 3),
        "midpoint_y": round(segment.midpoint_y, 3),
        "edge_confidence": round(segment.edge_confidence, 6),
        "edge_support_fraction": round(segment.edge_support_fraction, 6),
    }


def _edge_anchored_candidate_priority(segment: LineSegment, width: int, height: int) -> float:
    """Rank strong, real edge segments that are plausible depth-line evidence."""

    vertical_change_fraction = abs(segment.y2 - segment.y1) / max(segment.length, 1e-6)
    diagonal_weight = min(1.4, max(0.4, vertical_change_fraction * 3.0))
    outside_weight = 0.25 + 0.75 * _center_occluder_outside_fraction(segment, width, height)
    support_weight = 0.5 + segment.edge_support_fraction
    return segment.length * segment.edge_confidence * support_weight * diagonal_weight * outside_weight


def _edge_candidate_orientation_bin(angle_deg: float) -> int:
    """Return a coarse orientation bin for diversity among edge candidates."""

    return int(angle_deg // config.EDGE_ANCHORED_CANDIDATE_ORIENTATION_BIN_DEG)


def extract_edge_anchored_line_candidates(
    edge_probability_map: np.ndarray,
    frame_index: int,
) -> LineDetectionResult:
    """Extract finite line candidates strictly from the edge-probability tensor.

    These candidates are real line segments supported by sampled values in the
    edge tensor. Gemini may select candidate IDs from this list, but it does not
    provide geometry.
    """

    if edge_probability_map.ndim != 2 or edge_probability_map.size == 0:
        raise RuntimeError("Edge-anchored line extraction requires a non-empty 2D edge map.")
    if not hasattr(cv2, "createLineSegmentDetector"):
        raise RuntimeError(
            "OpenCV was built without LSD line segment detection. Install "
            "`opencv-python-headless` from requirements.txt."
        )

    height, width = edge_probability_map.shape
    image_diagonal = math.hypot(width, height)
    minimum_length = max(
        config.EDGE_ANCHORED_CANDIDATE_MIN_LENGTH_PX,
        config.EDGE_ANCHORED_CANDIDATE_MIN_LENGTH_IMAGE_FRACTION * image_diagonal,
    )

    edge_thresholded = (edge_probability_map >= config.EDGE_PROBABILITY_THRESHOLD).astype(np.uint8) * 255
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    detected_lsd = detector.detect(edge_thresholded)[0]
    detected_hough = cv2.HoughLinesP(
        edge_thresholded,
        rho=config.LINE_HOUGH_RHO_PX,
        theta=math.radians(config.LINE_HOUGH_THETA_DEG),
        threshold=config.EDGE_ANCHORED_CANDIDATE_HOUGH_THRESHOLD,
        minLineLength=minimum_length,
        maxLineGap=config.EDGE_ANCHORED_CANDIDATE_HOUGH_MAX_GAP_PX,
    )

    raw_lines: list[np.ndarray] = []
    if detected_lsd is not None:
        raw_lines.extend(detected_lsd.reshape(-1, 4))
    if detected_hough is not None:
        raw_lines.extend(detected_hough.reshape(-1, 4))
    if not raw_lines:
        return LineDetectionResult(frame_index=frame_index, accepted_segments=[], rejected_segments=[])

    provisional: list[LineSegment] = []
    raw_segment_id = 0
    for raw_line in raw_lines:
        x1, y1, x2, y2 = [float(value) for value in raw_line]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = _angle_deg(x1, y1, x2, y2)
        midpoint_x = 0.5 * (x1 + x2)
        midpoint_y = 0.5 * (y1 + y2)
        confidence, support_fraction = _edge_statistics(edge_probability_map, (x1, y1, x2, y2))
        segment = LineSegment(
            segment_id=raw_segment_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            length=length,
            angle_deg=angle,
            midpoint_x=midpoint_x,
            midpoint_y=midpoint_y,
            edge_confidence=confidence,
            edge_support_fraction=support_fraction,
            accepted=False,
            rejection_reason=None,
            family_label=None,
        )
        raw_segment_id += 1

        if length < minimum_length:
            continue
        if abs(y2 - y1) / length < config.EDGE_ANCHORED_CANDIDATE_MIN_VERTICAL_CHANGE_FRACTION:
            continue
        if _orientation_delta_deg(angle, 90.0) < config.EDGE_ANCHORED_CANDIDATE_VERTICAL_REJECTION_ANGLE_DEG:
            continue
        if confidence < config.EDGE_ANCHORED_CANDIDATE_MIN_EDGE_CONFIDENCE:
            continue
        if support_fraction < config.EDGE_ANCHORED_CANDIDATE_MIN_EDGE_SUPPORT_FRACTION:
            continue
        if _is_frame_border_line(segment, width, height):
            continue
        if (
            _center_occluder_outside_fraction(segment, width, height)
            < config.EDGE_ANCHORED_CANDIDATE_CENTER_OCCLUDER_MIN_OUTSIDE_FRACTION
        ):
            continue
        provisional.append(segment)

    deduplicated: list[LineSegment] = []
    for segment in sorted(
        provisional,
        key=lambda item: (-_edge_anchored_candidate_priority(item, width, height), item.segment_id),
    ):
        if _is_near_duplicate(segment, deduplicated):
            continue
        deduplicated.append(segment)

    accepted: list[LineSegment] = []
    orientation_counts: dict[int, int] = {}
    for segment in deduplicated:
        orientation_bin = _edge_candidate_orientation_bin(segment.angle_deg)
        if orientation_counts.get(orientation_bin, 0) >= config.EDGE_ANCHORED_CANDIDATE_MAX_PER_ORIENTATION_BIN:
            continue
        orientation_counts[orientation_bin] = orientation_counts.get(orientation_bin, 0) + 1
        accepted.append(
            replace(
                segment,
                segment_id=len(accepted),
                accepted=True,
                rejection_reason=None,
                family_label="edge_tensor_candidate",
            )
        )
        if len(accepted) >= config.EDGE_ANCHORED_CANDIDATE_MAX_PER_FRAME:
            break

    return LineDetectionResult(frame_index=frame_index, accepted_segments=accepted, rejected_segments=[])


def extract_line_segments(
    edge_probability_map: np.ndarray,
    frame_index: int,
) -> LineDetectionResult:
    """Detect LSD segments over the edge probability map and filter them."""

    if edge_probability_map.ndim != 2 or edge_probability_map.size == 0:
        raise RuntimeError("Line extraction requires a non-empty 2D edge probability map.")
    if not hasattr(cv2, "createLineSegmentDetector"):
        raise RuntimeError(
            "OpenCV was built without LSD line segment detection. Install "
            "`opencv-python-headless` from requirements.txt."
        )

    height, width = edge_probability_map.shape
    image_diagonal = math.hypot(width, height)
    minimum_length = max(
        config.LINE_MIN_LENGTH_PX,
        config.LINE_MIN_LENGTH_IMAGE_FRACTION * image_diagonal,
    )

    # LSD is run on the DexiNed probability image itself, not on Canny edges.
    edge_image = np.round(edge_probability_map * 255.0).astype(np.uint8)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    detected_lsd = detector.detect(edge_image)[0]

    # Probabilistic Hough is added over the thresholded DexiNed map to recover
    # long straight structures when LSD is overly conservative.
    edge_thresholded = (edge_probability_map >= config.EDGE_PROBABILITY_THRESHOLD).astype(np.uint8) * 255
    detected_hough = cv2.HoughLinesP(
        edge_thresholded,
        rho=config.LINE_HOUGH_RHO_PX,
        theta=math.radians(config.LINE_HOUGH_THETA_DEG),
        threshold=config.LINE_HOUGH_VOTE_THRESHOLD,
        minLineLength=minimum_length,
        maxLineGap=config.LINE_HOUGH_MAX_LINE_GAP_PX,
    )

    raw_lines: list[np.ndarray] = []
    if detected_lsd is not None:
        raw_lines.extend(detected_lsd.reshape(-1, 4))
    if detected_hough is not None:
        raw_lines.extend(detected_hough.reshape(-1, 4))
    if not raw_lines:
        return LineDetectionResult(frame_index=frame_index, accepted_segments=[], rejected_segments=[])

    provisional: list[LineSegment] = []
    rejected: list[LineSegment] = []
    segment_id = 0
    for raw_line in raw_lines:
        x1, y1, x2, y2 = [float(value) for value in raw_line]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = _angle_deg(x1, y1, x2, y2)
        midpoint_x = 0.5 * (x1 + x2)
        midpoint_y = 0.5 * (y1 + y2)
        confidence, support_fraction = _edge_statistics(edge_probability_map, (x1, y1, x2, y2))
        segment = LineSegment(
            segment_id=segment_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            length=length,
            angle_deg=angle,
            midpoint_x=midpoint_x,
            midpoint_y=midpoint_y,
            edge_confidence=confidence,
            edge_support_fraction=support_fraction,
            accepted=False,
            rejection_reason=None,
            family_label=None,
        )
        segment_id += 1

        if length < minimum_length:
            segment.rejection_reason = "too_short"
            rejected.append(segment)
            continue
        if abs(y2 - y1) / length < config.LINE_MIN_VERTICAL_CHANGE_FRACTION:
            segment.rejection_reason = "near_frontal_horizontal"
            rejected.append(segment)
            continue
        if _orientation_delta_deg(angle, 90.0) < config.LINE_VERTICAL_REJECTION_ANGLE_DEG:
            segment.rejection_reason = "near_vertical"
            rejected.append(segment)
            continue
        if confidence < config.LINE_MIN_EDGE_CONFIDENCE:
            segment.rejection_reason = "low_edge_confidence"
            rejected.append(segment)
            continue
        if support_fraction < config.LINE_MIN_EDGE_SUPPORT_FRACTION:
            segment.rejection_reason = "low_edge_support_fraction"
            rejected.append(segment)
            continue
        if _is_frame_border_line(segment, width, height):
            segment.rejection_reason = "image_border"
            rejected.append(segment)
            continue
        if (
            _center_occluder_outside_fraction(segment, width, height)
            < config.LINE_CENTER_OCCLUDER_MIN_OUTSIDE_FRACTION
        ):
            segment.rejection_reason = "central_foreground_occluder"
            rejected.append(segment)
            continue
        if (
            midpoint_y < config.LINE_SHORT_TEXTURE_UPPER_REGION_FRACTION * height
            and length < config.LINE_SHORT_TEXTURE_LENGTH_MULTIPLIER * minimum_length
        ):
            segment.rejection_reason = "upper_region_short_texture"
            rejected.append(segment)
            continue
        if not _is_ground_plane_parallel_candidate(segment, height, minimum_length):
            segment.rejection_reason = "not_ground_plane_parallel_candidate"
            rejected.append(segment)
            continue
        provisional.append(segment)

    kept: list[LineSegment] = []
    for segment in sorted(provisional, key=lambda item: (-item.length * item.edge_confidence, item.segment_id)):
        if _is_near_duplicate(segment, kept):
            segment.rejection_reason = "near_duplicate"
            rejected.append(segment)
            continue
        kept.append(segment)

    selected_families = _select_orientation_families(kept, image_diagonal)
    family_candidates: list[LineSegment] = []
    for segment in kept:
        label = _family_label(segment.angle_deg)
        if label not in selected_families:
            segment.rejection_reason = "weak_orientation_family"
            rejected.append(segment)
            continue
        segment.family_label = label
        family_candidates.append(segment)

    accepted: list[LineSegment] = []
    ranked_family_candidates = sorted(
        family_candidates,
        key=lambda item: (-_ground_plane_priority(item, width, height), item.segment_id),
    )
    for index, segment in enumerate(ranked_family_candidates):
        if index >= config.LINE_MAX_ACCEPTED_SEGMENTS_PER_FRAME:
            segment.rejection_reason = "low_ground_plane_priority"
            rejected.append(segment)
            continue
        segment.accepted = True
        accepted.append(segment)

    accepted.sort(key=lambda item: item.segment_id)
    rejected.sort(key=lambda item: item.segment_id)
    return LineDetectionResult(
        frame_index=frame_index,
        accepted_segments=accepted,
        rejected_segments=rejected,
    )


def line_result_from_gemini_selection(
    frame_index: int,
    selected_lines: list[dict[str, object]],
) -> LineDetectionResult:
    """Convert Gemini-selected finite edge endpoints into accepted line segments."""

    accepted: list[LineSegment] = []
    for segment_id, line in enumerate(selected_lines):
        x1 = float(line["x1"])
        y1 = float(line["y1"])
        x2 = float(line["x2"])
        y2 = float(line["y2"])
        length = math.hypot(x2 - x1, y2 - y1)
        angle = _angle_deg(x1, y1, x2, y2)
        confidence = float(line.get("confidence", 1.0))
        rationale = str(
            line.get(
                "why_this_projects_away_from_camera",
                line.get("why_this_is_ground_parallel", ""),
            )
        )
        accepted.append(
            LineSegment(
                segment_id=segment_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                length=length,
                angle_deg=angle,
                midpoint_x=0.5 * (x1 + x2),
                midpoint_y=0.5 * (y1 + y2),
                edge_confidence=confidence,
                edge_support_fraction=1.0,
                accepted=True,
                rejection_reason=None,
                family_label="gemini_receding_depth_parallel",
                object_or_scene_edge=str(line.get("object_or_scene_edge", "")),
                selection_rationale=rationale,
                selection_confidence=confidence,
            )
        )
    return LineDetectionResult(frame_index=frame_index, accepted_segments=accepted, rejected_segments=[])


def line_result_from_edge_candidate_selection(
    frame_index: int,
    candidate_result: LineDetectionResult,
    selection: dict[str, object],
) -> LineDetectionResult:
    """Convert selected edge-candidate IDs into accepted VP line segments."""

    selected_ids = {
        str(candidate_id)
        for candidate_id in selection.get("selected_candidate_ids", [])
        if isinstance(candidate_id, str)
    }
    selected_metadata = {
        str(line.get("candidate_id")): line
        for line in selection.get("lines", [])
        if isinstance(line, dict) and line.get("candidate_id") is not None
    }
    usable = bool(selection.get("usable"))
    accepted: list[LineSegment] = []
    rejected: list[LineSegment] = []

    for candidate in candidate_result.accepted_segments:
        candidate_id = line_candidate_id(candidate)
        if usable and candidate_id in selected_ids:
            metadata = selected_metadata.get(candidate_id, {})
            confidence = float(metadata.get("confidence", candidate.edge_confidence))
            accepted.append(
                replace(
                    candidate,
                    segment_id=len(accepted),
                    accepted=True,
                    rejection_reason=None,
                    family_label="edge_tensor_gemini_receding_depth",
                    object_or_scene_edge=str(metadata.get("object_or_scene_edge", "edge tensor candidate")),
                    selection_rationale=str(
                        metadata.get(
                            "why_this_projects_away_from_camera",
                            "Gemini selected this real edge candidate ID as receding-depth evidence.",
                        )
                    ),
                    selection_confidence=confidence,
                )
            )
        else:
            rejected.append(
                replace(
                    candidate,
                    segment_id=len(rejected),
                    accepted=False,
                    rejection_reason="not_selected_as_receding_depth_candidate",
                    family_label="edge_tensor_candidate",
                )
            )

    return LineDetectionResult(
        frame_index=frame_index,
        accepted_segments=accepted,
        rejected_segments=rejected,
    )
