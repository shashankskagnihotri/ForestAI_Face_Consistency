"""Writers for benchmark metadata, JSON results, and markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config
from .frame_sampling import SampledFrame
from .schemas import BinaryLabel
from .utils import display_path, write_json
from .video_io import VideoMetadata


def write_metadata(metadata: VideoMetadata, runtime_config: config.RuntimeConfig) -> dict[str, Any]:
    """Write metadata.json with the required video and sampling fields."""

    payload: dict[str, Any] = {
        "video_path": display_path(metadata.video_path, runtime_config.project_root),
        "video_filename": metadata.filename,
        "video_duration_seconds": round(metadata.duration_seconds, 2),
        "frame_count": metadata.frame_count,
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "sampling_rule": (
            f"Every {config.FRAME_SAMPLING_STRIDE}th frame plus "
            f"{config.NUM_RANDOM_FRAMES} uniformly random frames with deduplication."
        ),
        "random_seed": config.RANDOM_SEED,
        "created_result_directory": display_path(runtime_config.result_dir, runtime_config.project_root),
        "timestamp_source": metadata.timestamp_source,
    }
    write_json(runtime_config.metadata_path, payload)
    return payload


def write_results_json(
    metadata: VideoMetadata,
    runtime_config: config.RuntimeConfig,
    prompt_for_video: str,
    sampled_frames: list[SampledFrame],
    vanishing_point_consistency: BinaryLabel,
    single_light_source_consistency: BinaryLabel,
    prompt_object_recognizability: BinaryLabel,
) -> dict[str, Any]:
    """Write the final required results.json structure."""

    payload: dict[str, Any] = {
        "video_path": display_path(metadata.video_path, runtime_config.project_root),
        "prompt_for_video": prompt_for_video,
        "model": config.GEMINI_MODEL_NAME,
        "video_duration_seconds": round(metadata.duration_seconds, 2),
        "result_dir": display_path(runtime_config.result_dir, runtime_config.project_root),
        "sampled_frame_indices": [frame.frame_index for frame in sampled_frames],
        "sampled_frame_timestamps_sec": [round(frame.timestamp_sec, 6) for frame in sampled_frames],
        "vanishing_point_consistency": vanishing_point_consistency,
        "single_light_source_consistency": single_light_source_consistency,
        "prompt_object_recognizability": prompt_object_recognizability,
        "metadata_path": display_path(runtime_config.metadata_path, runtime_config.project_root),
        "vanishing_point_diagnostics_path": display_path(
            runtime_config.vanishing_point_diagnostics_path,
            runtime_config.project_root,
        ),
        "sampled_frames_dir": display_path(runtime_config.sampled_frames_dir, runtime_config.project_root),
        "edge_maps_dir": display_path(runtime_config.edge_maps_dir, runtime_config.project_root),
        "annotated_frames_dir": display_path(runtime_config.annotated_frames_dir, runtime_config.project_root),
        "gemini_ground_parallel_line_selections_path": display_path(
            runtime_config.gemini_ground_parallel_line_selections_path,
            runtime_config.project_root,
        ),
        "single_light_source_analysis_path": display_path(
            runtime_config.gemini_responses_dir / "single_light_source_response.json",
            runtime_config.project_root,
        ),
        "prompt_object_visibility_analysis_path": display_path(
            runtime_config.gemini_responses_dir / "prompt_object_recognizability_response.json",
            runtime_config.project_root,
        ),
        "contact_sheet_path": display_path(runtime_config.contact_sheet_path, runtime_config.project_root),
        "gemini_requests_dir": display_path(runtime_config.gemini_requests_dir, runtime_config.project_root),
        "gemini_responses_dir": display_path(runtime_config.gemini_responses_dir, runtime_config.project_root),
        "benchmark_report_path": display_path(runtime_config.benchmark_report_path, runtime_config.project_root),
    }
    write_json(runtime_config.results_json_path, payload)
    return payload


def write_benchmark_report(
    metadata: VideoMetadata,
    runtime_config: config.RuntimeConfig,
    prompt_for_video: str,
    sampled_frames: list[SampledFrame],
    vanishing_point_consistency: BinaryLabel,
    single_light_source_consistency: BinaryLabel,
    prompt_object_recognizability: BinaryLabel,
) -> Path:
    """Write a concise benchmark report with no chain-of-thought."""

    sampled_indices = [frame.frame_index for frame in sampled_frames]
    sampled_timestamps = [round(frame.timestamp_sec, 6) for frame in sampled_frames]
    lines = [
        "# Video Realism Benchmark Report",
        "",
        "## Input",
        f"- Video path: {display_path(metadata.video_path, runtime_config.project_root)}",
        f"- Video duration: {metadata.duration_seconds:.2f} seconds",
        f"- Prompt used to generate the video: {prompt_for_video}",
        "",
        "## Sampled Frames",
        f"- Frame indices: {sampled_indices}",
        f"- Frame timestamps, seconds: {sampled_timestamps}",
        "",
        "## Final Binary Results",
        f"- Vanishing point consistency: {vanishing_point_consistency}",
        f"- Single light source consistency: {single_light_source_consistency}",
        f"- Prompt-object recognizability: {prompt_object_recognizability}",
        "",
        "## Generated Artifacts",
        f"- Raw sampled frames: {display_path(runtime_config.sampled_frames_dir, runtime_config.project_root)}",
        f"- DexiNed edge probability maps: {display_path(runtime_config.edge_maps_dir, runtime_config.project_root)}",
        f"- Edge-tensor-derived receding/depth-parallel VP line evidence: {display_path(runtime_config.gemini_ground_parallel_line_selections_path, runtime_config.project_root)}",
        f"- Annotated vanishing-point visualizations: {display_path(runtime_config.annotated_frames_dir, runtime_config.project_root)}",
        f"- Vanishing-point diagnostics JSON: {display_path(runtime_config.vanishing_point_diagnostics_path, runtime_config.project_root)}",
        f"- Contact sheet: {display_path(runtime_config.contact_sheet_path, runtime_config.project_root)}",
        f"- Gemini request metadata: {display_path(runtime_config.gemini_requests_dir, runtime_config.project_root)}",
        f"- Compact Gemini structured responses: {display_path(runtime_config.gemini_responses_dir, runtime_config.project_root)}",
        f"- Final JSON results: {display_path(runtime_config.results_json_path, runtime_config.project_root)}",
        "",
        "## Limitations",
        "- The VP line evidence uses full-resolution edge-tensor line extraction plus Gemini candidate-ID selection and Lu-style spherical voting; it still depends on visible straight scene/object edges being present and recognizable.",
        "- The vanishing-point estimator provides evidence and diagnostics; the final yes/no answer is the strict Gemini binary judgment, not a mathematical proof of scene geometry.",
        "- Insufficient visual evidence is handled conservatively by the Gemini prompt and should produce `no`.",
        "- This report contains no chain-of-thought or hidden reasoning.",
        "",
    ]
    runtime_config.benchmark_report_path.write_text("\n".join(lines), encoding="utf-8")
    return runtime_config.benchmark_report_path
