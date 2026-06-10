"""Central configuration for the video realism benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


PROJECT_NAME: Final[str] = "video_realism_benchmark"

# Gemini configuration. The benchmark must not silently switch this model.
GEMINI_MODEL_NAME: Final[str] = "gemini-3.1-pro-preview"
DEFAULT_GEMINI_API_KEY_ENV_VAR: Final[str] = "GEMINI_ForestAI_API_KEY"
ALLOWED_BINARY_LABELS: Final[tuple[str, str]] = ("yes", "no")
GEMINI_HTTP_TIMEOUT_MS: Final[int] = 120_000
GEMINI_REQUEST_MAX_ATTEMPTS: Final[int] = 2
GEMINI_JSON_PARSE_MAX_ATTEMPTS: Final[int] = 2
GEMINI_REQUEST_RETRY_SLEEP_SEC: Final[float] = 5.0
GEMINI_BINARY_MAX_OUTPUT_TOKENS: Final[int] = 256
GEMINI_JSON_MAX_OUTPUT_TOKENS: Final[int] = 8192
GEMINI_FILE_PROCESSING_TIMEOUT_SEC: Final[float] = 600.0
GEMINI_FILE_POLL_INTERVAL_SEC: Final[float] = 2.0
GEMINI_LINE_SELECTION_MAX_LINES_PER_FRAME: Final[int] = 12
GEMINI_LINE_SELECTION_MIN_CONFIDENCE: Final[float] = 0.35
GEMINI_LINE_SELECTION_MIN_LENGTH_PX: Final[float] = 35.0
GEMINI_LINE_SELECTION_MIN_VERTICAL_CHANGE_FRACTION: Final[float] = 0.045
GEMINI_LINE_SELECTION_VERTICAL_REJECTION_ANGLE_DEG: Final[float] = 12.0
GEMINI_LINE_SELECTION_CENTER_OCCLUDER_MIN_OUTSIDE_FRACTION: Final[float] = 0.20
GEMINI_LINE_SELECTION_BORDER_MARGIN_PX: Final[float] = 4.0
LIGHT_SOURCE_DIRECTIONS: Final[tuple[str, ...]] = (
    "center",
    "upper",
    "lower",
    "left",
    "right",
    "upper-left",
    "upper-right",
    "lower-left",
    "lower-right",
    "unknown",
)

# Deterministic frame sampling.
RANDOM_SEED: Final[int] = 1729
FRAME_SAMPLING_STRIDE: Final[int] = 16
NUM_RANDOM_FRAMES: Final[int] = 3

# Output layout. Benchmark artifacts go under results only.
RESULTS_DIR_NAME: Final[str] = "results"
DEBUGGING_DIR_NAME: Final[str] = "debugging"
FIRST_REPORT_FILENAME: Final[str] = "first_report.md"
SAMPLE_REPORT_FILENAME: Final[str] = "sample.md"
NEXT_TASK_FILENAME: Final[str] = "next_task.txt"
ALLOWED_DEBUGGING_FILENAMES: Final[tuple[str, ...]] = (
    FIRST_REPORT_FILENAME,
    SAMPLE_REPORT_FILENAME,
    NEXT_TASK_FILENAME,
)
ALLOWED_DEBUGGING_NOTE_SUFFIXES: Final[tuple[str, ...]] = (".md", ".txt")
SAMPLED_FRAMES_DIRNAME: Final[str] = "sampled_frames"
EDGE_MAPS_DIRNAME: Final[str] = "edge_maps"
ANNOTATED_FRAMES_DIRNAME: Final[str] = "annotated_frames"
GEMINI_REQUESTS_DIRNAME: Final[str] = "gemini_requests"
GEMINI_RESPONSES_DIRNAME: Final[str] = "gemini_responses"

# DexiNed edge detector. The code requires real weights and downloads the
# default checkpoint on first run when it is not already present.
EDGE_DETECTOR_NAME: Final[str] = "DexiNed"
DEXINED_WEIGHTS_ENV_VAR: Final[str] = "DEXINED_WEIGHTS_PATH"
DEXINED_HF_REPO_ENV_VAR: Final[str] = "DEXINED_HF_REPO_ID"
DEXINED_HF_FILENAME_ENV_VAR: Final[str] = "DEXINED_HF_FILENAME"
DEFAULT_DEXINED_HF_REPO_ID: Final[str] = "kornia/dexined"
DEFAULT_DEXINED_HF_FILENAME: Final[str] = "DexiNed_BIPED_10.pth"
DEFAULT_DEXINED_WEIGHTS_PATH: Final[Path] = Path("models") / "DexiNed_BIPED_10.pth"
EDGE_DETECTOR_DEVICE: Final[str] = "cpu"
EDGE_INPUT_MAX_LONG_SIDE: Final[int] = 1280
EDGE_PROBABILITY_THRESHOLD: Final[float] = 0.18
EDGE_NORMALIZATION_LOWER_PERCENTILE: Final[float] = 50.0
EDGE_NORMALIZATION_UPPER_PERCENTILE: Final[float] = 99.8
EDGE_NORMALIZATION_EPS: Final[float] = 1e-6

# Line extraction thresholds. These values are intentionally centralized.
LINE_MIN_LENGTH_PX: Final[float] = 40.0
LINE_MIN_LENGTH_IMAGE_FRACTION: Final[float] = 0.04
LINE_MIN_EDGE_CONFIDENCE: Final[float] = 0.12
LINE_MIN_EDGE_SUPPORT_FRACTION: Final[float] = 0.35
LINE_DUPLICATE_ANGLE_DEG: Final[float] = 4.0
LINE_DUPLICATE_MIDPOINT_PX: Final[float] = 20.0
LINE_DUPLICATE_ENDPOINT_PX: Final[float] = 35.0
LINE_BORDER_MARGIN_FRACTION: Final[float] = 0.008
LINE_SHORT_TEXTURE_UPPER_REGION_FRACTION: Final[float] = 0.30
LINE_SHORT_TEXTURE_LENGTH_MULTIPLIER: Final[float] = 1.50
LINE_MIN_VERTICAL_CHANGE_FRACTION: Final[float] = 0.045
LINE_VERTICAL_REJECTION_ANGLE_DEG: Final[float] = 12.0
LINE_GROUND_PLANE_MIN_MIDPOINT_Y_FRACTION: Final[float] = 0.38
LINE_GROUND_PLANE_MIN_LOWER_ENDPOINT_Y_FRACTION: Final[float] = 0.50
LINE_GROUND_PLANE_LONG_LINE_MULTIPLIER: Final[float] = 1.20
LINE_CENTER_OCCLUDER_X_MIN_FRACTION: Final[float] = 0.30
LINE_CENTER_OCCLUDER_X_MAX_FRACTION: Final[float] = 0.70
LINE_CENTER_OCCLUDER_Y_MIN_FRACTION: Final[float] = 0.22
LINE_CENTER_OCCLUDER_Y_MAX_FRACTION: Final[float] = 0.98
LINE_CENTER_OCCLUDER_MIN_OUTSIDE_FRACTION: Final[float] = 0.20
LINE_ORIENTATION_BIN_DEG: Final[float] = 10.0
LINE_MIN_ORIENTATION_FAMILY_COUNT: Final[int] = 2
LINE_MIN_ORIENTATION_FAMILY_SUPPORT_FRACTION: Final[float] = 0.06
LINE_MAX_STRUCTURAL_FAMILIES: Final[int] = 4
LINE_MAX_ACCEPTED_SEGMENTS_PER_FRAME: Final[int] = 48
LINE_CONFIDENCE_SAMPLE_COUNT: Final[int] = 64
LINE_HOUGH_RHO_PX: Final[float] = 1.0
LINE_HOUGH_THETA_DEG: Final[float] = 1.0
LINE_HOUGH_VOTE_THRESHOLD: Final[int] = 24
LINE_HOUGH_MAX_LINE_GAP_PX: Final[int] = 10
EDGE_ANCHORED_CANDIDATE_MAX_PER_FRAME: Final[int] = 36
EDGE_ANCHORED_CANDIDATE_MIN_LENGTH_PX: Final[float] = 55.0
EDGE_ANCHORED_CANDIDATE_MIN_LENGTH_IMAGE_FRACTION: Final[float] = 0.045
EDGE_ANCHORED_CANDIDATE_MIN_EDGE_CONFIDENCE: Final[float] = 0.18
EDGE_ANCHORED_CANDIDATE_MIN_EDGE_SUPPORT_FRACTION: Final[float] = 0.55
EDGE_ANCHORED_CANDIDATE_MIN_VERTICAL_CHANGE_FRACTION: Final[float] = 0.055
EDGE_ANCHORED_CANDIDATE_VERTICAL_REJECTION_ANGLE_DEG: Final[float] = 12.0
EDGE_ANCHORED_CANDIDATE_CENTER_OCCLUDER_MIN_OUTSIDE_FRACTION: Final[float] = 0.35
EDGE_ANCHORED_CANDIDATE_HOUGH_THRESHOLD: Final[int] = 32
EDGE_ANCHORED_CANDIDATE_HOUGH_MAX_GAP_PX: Final[int] = 6
EDGE_ANCHORED_CANDIDATE_ORIENTATION_BIN_DEG: Final[float] = 8.0
EDGE_ANCHORED_CANDIDATE_MAX_PER_ORIENTATION_BIN: Final[int] = 5

# Vanishing-point robust estimation thresholds.
VP_RANSAC_ITERATIONS: Final[int] = 2000
VP_RANSAC_ANGULAR_THRESHOLD_DEG: Final[float] = 3.0
VP_MIN_INLIER_LINES: Final[int] = 2
VP_MIN_INLIER_RATIO: Final[float] = 0.35
VP_PARALLEL_INTERSECTION_EPS: Final[float] = 1e-7
VP_MAX_COORD_MULTIPLIER: Final[float] = 50.0
VP_EXHAUSTIVE_PAIR_LIMIT: Final[int] = 6000
VP_LU_RANSAC_ITERATIONS: Final[int] = 120
VP_LU_SPHERE_BIN_DEG: Final[float] = 1.0
VP_LU_PAIR_ORIENTATION_MAX_DEG: Final[float] = 60.0
VP_LU_CLUSTER_ANGULAR_THRESHOLD_DEG: Final[float] = 6.0
VP_LU_FOCAL_LENGTH_MULTIPLIER: Final[float] = 1.2
VP_LU_VERTICAL_REJECTION_Y_MULTIPLIER: Final[float] = 3.0
VP_REFERENCE_ANGULAR_THRESHOLD_DEG: Final[float] = 6.0
VP_REFERENCE_MIN_INLIER_RATIO: Final[float] = 0.22

# Visualization settings.
VIS_CONTACT_SHEET_THUMB_WIDTH: Final[int] = 420
VIS_CONTACT_SHEET_MAX_COLUMNS: Final[int] = 4
VIS_LINE_THICKNESS: Final[int] = 2
VIS_EXTENDED_LINE_THICKNESS: Final[int] = 1
VP_VISUAL_STRICT_INLIER_THRESHOLD_DEG: Final[float] = 1.0
VIS_FONT_SCALE: Final[float] = 0.55
VIS_FONT_THICKNESS: Final[int] = 1


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved paths and constants for one benchmark run."""

    project_root: Path
    result_dir: Path
    sampled_frames_dir: Path
    edge_maps_dir: Path
    annotated_frames_dir: Path
    gemini_requests_dir: Path
    gemini_responses_dir: Path
    results_json_path: Path
    benchmark_report_path: Path
    metadata_path: Path
    vanishing_point_diagnostics_path: Path
    gemini_ground_parallel_line_selections_path: Path
    contact_sheet_path: Path
    model_name: str = GEMINI_MODEL_NAME
    random_seed: int = RANDOM_SEED
