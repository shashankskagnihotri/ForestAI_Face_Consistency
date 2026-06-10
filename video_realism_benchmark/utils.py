"""Shared path, JSON, logging, and validation helpers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from . import config


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure quiet logging so successful CLI output remains exact."""

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def sanitize_video_stem(stem: str) -> str:
    """Return a filesystem-safe, deterministic video stem."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "video"


def format_duration_seconds(seconds: float) -> str:
    """Format a duration with the exact precision required in result paths."""

    return f"{seconds:.2f}"


def display_path(path: Path, project_root: Path) -> str:
    """Return a stable display path, preferring ./relative paths."""

    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return f"./{resolved.relative_to(root).as_posix()}"
    except ValueError:
        return resolved.as_posix()


def make_unique_result_dir(project_root: Path, safe_video_name: str, duration_seconds: float) -> Path:
    """Create the per-video result directory without silent overwrites."""

    results_root = project_root / config.RESULTS_DIR_NAME
    results_root.mkdir(parents=True, exist_ok=True)

    duration_text = format_duration_seconds(duration_seconds)
    base_name = f"{safe_video_name}_{duration_text}s"
    candidate = results_root / base_name
    if not candidate.exists():
        candidate.mkdir(parents=False)
        return candidate

    run_index = 1
    while True:
        suffixed = results_root / f"{base_name}_run_{run_index:03d}"
        if not suffixed.exists():
            suffixed.mkdir(parents=False)
            return suffixed
        run_index += 1


def build_runtime_config(project_root: Path, result_dir: Path) -> config.RuntimeConfig:
    """Create all benchmark subdirectories and return the resolved config."""

    sampled_frames_dir = result_dir / config.SAMPLED_FRAMES_DIRNAME
    edge_maps_dir = result_dir / config.EDGE_MAPS_DIRNAME
    annotated_frames_dir = result_dir / config.ANNOTATED_FRAMES_DIRNAME
    gemini_requests_dir = result_dir / config.GEMINI_REQUESTS_DIRNAME
    gemini_responses_dir = result_dir / config.GEMINI_RESPONSES_DIRNAME

    for directory in (
        sampled_frames_dir,
        edge_maps_dir,
        annotated_frames_dir,
        gemini_requests_dir,
        gemini_responses_dir,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    return config.RuntimeConfig(
        project_root=project_root,
        result_dir=result_dir,
        sampled_frames_dir=sampled_frames_dir,
        edge_maps_dir=edge_maps_dir,
        annotated_frames_dir=annotated_frames_dir,
        gemini_requests_dir=gemini_requests_dir,
        gemini_responses_dir=gemini_responses_dir,
        results_json_path=result_dir / "results.json",
        benchmark_report_path=result_dir / "benchmark_report.md",
        metadata_path=result_dir / "metadata.json",
        vanishing_point_diagnostics_path=result_dir / "vanishing_point_diagnostics.json",
        gemini_ground_parallel_line_selections_path=result_dir / "gemini_ground_parallel_line_selections.json",
        contact_sheet_path=result_dir / "contact_sheet_vanishing_point.png",
    )


def enforce_debugging_directory_policy(project_root: Path) -> None:
    """Ensure ./debugging contains only lightweight human debugging notes."""

    debugging_dir = project_root / config.DEBUGGING_DIR_NAME
    debugging_dir.mkdir(parents=True, exist_ok=True)
    allowed_names = set(config.ALLOWED_DEBUGGING_FILENAMES)
    allowed_suffixes = set(config.ALLOWED_DEBUGGING_NOTE_SUFFIXES)
    unexpected = [
        p
        for p in debugging_dir.iterdir()
        if not (
            (p.name in allowed_names and p.is_file())
            or (p.is_file() and p.suffix in allowed_suffixes)
            or _is_prompt_debug_note_directory(p, allowed_suffixes)
        )
    ]
    if unexpected:
        names = ", ".join(p.name for p in unexpected)
        raise RuntimeError(
            f"./{config.DEBUGGING_DIR_NAME} may contain only lightweight debug notes "
            f"{sorted(allowed_names)} and prompt-specific note directories; "
            f"unexpected entries: {names}"
        )


def _is_prompt_debug_note_directory(path: Path, allowed_suffixes: set[str]) -> bool:
    """Allow lightweight prompt-specific reports such as ./debugging/prompt_2/*.md."""

    if not path.is_dir() or not re.fullmatch(r"prompt_[A-Za-z0-9_-]+", path.name):
        return False
    for child in path.rglob("*"):
        if child.is_dir():
            continue
        if not child.is_file() or child.suffix not in allowed_suffixes:
            return False
    return True


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic, readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)
        file.write("\n")


def assert_file_exists(path: Path, description: str) -> None:
    """Fail loudly if a required file is missing."""

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required {description} does not exist: {path}")
