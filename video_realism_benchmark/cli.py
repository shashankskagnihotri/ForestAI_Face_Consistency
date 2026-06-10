"""Argparse CLI for the video realism benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import config
from .edge_detection import DexiNedEdgeDetector
from .frame_sampling import extract_sampled_frames, sample_frame_indices
from .gemini_client import (
    GeminiEvidence,
    GeminiJudgeClient,
    evaluate_prompt_object_recognizability,
    evaluate_single_light_source,
    evaluate_vanishing_point,
    select_receding_depth_lines,
)
from .line_detection import (
    extract_edge_anchored_line_candidates,
    line_result_from_edge_candidate_selection,
)
from .report_writer import write_benchmark_report, write_metadata, write_results_json
from .utils import (
    build_runtime_config,
    configure_logging,
    display_path,
    enforce_debugging_directory_policy,
    make_unique_result_dir,
    sanitize_video_stem,
    write_json,
)
from .vanishing_point import (
    estimate_frame_vanishing_point,
    estimate_video_global_vanishing_point,
    make_frame_diagnostics,
)
from .video_io import probe_video
from .visualization import create_annotated_frame, create_contact_sheet, create_edge_candidate_selection_frame


def build_parser() -> argparse.ArgumentParser:
    """Create the required CLI parser."""

    parser = argparse.ArgumentParser(prog=config.PROJECT_NAME)
    parser.add_argument("--video_path", required=True, help="Path to the video file to evaluate.")
    parser.add_argument(
        "--prompt_for_video",
        required=True,
        help="The text prompt that was used to generate the video.",
    )
    parser.add_argument(
        "--google_cloud_API_key_name",
        default=config.DEFAULT_GEMINI_API_KEY_ENV_VAR,
        help=(
            "Name of the environment variable containing the Gemini / Google API key. "
            f"Defaults to {config.DEFAULT_GEMINI_API_KEY_ENV_VAR}."
        ),
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> tuple[str, str, str, Path, Path]:
    """Run the full deterministic preprocessing and Gemini binary judging pipeline."""

    project_root = Path.cwd().resolve()
    enforce_debugging_directory_policy(project_root)

    # The API key must be read from the environment variable named by the user.
    try:
        api_key = os.environ[args.google_cloud_API_key_name]
    except KeyError as exc:
        raise RuntimeError(
            f"Environment variable {args.google_cloud_API_key_name!r} is not set; "
            "it must contain the Gemini / Google API key."
        ) from exc

    if not args.prompt_for_video.strip():
        raise RuntimeError("--prompt_for_video must not be empty.")

    video_path = Path(args.video_path).expanduser()
    metadata = probe_video(video_path)

    safe_video_name = sanitize_video_stem(metadata.video_path.stem)
    result_dir = make_unique_result_dir(project_root, safe_video_name, metadata.duration_seconds)
    runtime_config = build_runtime_config(project_root, result_dir)
    write_metadata(metadata, runtime_config)

    # Sample, decode, and save raw frame evidence.
    sampled_indices = sample_frame_indices(metadata.frame_count)
    sampled_frames = extract_sampled_frames(
        metadata.video_path,
        sampled_indices,
        metadata.frame_timestamps_sec,
        runtime_config.sampled_frames_dir,
    )

    # Run the required pretrained deep edge detector once per sampled frame.
    edge_detector = DexiNedEdgeDetector()
    edge_results = edge_detector.run(sampled_frames, runtime_config.edge_maps_dir)

    gemini_client = GeminiJudgeClient(api_key=api_key, runtime_config=runtime_config)
    edge_candidate_results = [
        extract_edge_anchored_line_candidates(edge_result.probability_map, sampled_frame.frame_index)
        for sampled_frame, edge_result in zip(sampled_frames, edge_results, strict=True)
    ]
    edge_candidate_overlay_paths = [
        create_edge_candidate_selection_frame(
            sampled_frame,
            edge_result,
            candidate_result,
            runtime_config.annotated_frames_dir / f"frame_{sampled_frame.frame_index:06d}_edge_candidates.png",
        )
        for sampled_frame, edge_result, candidate_result in zip(
            sampled_frames,
            edge_results,
            edge_candidate_results,
            strict=True,
        )
    ]
    gemini_line_selections = [
        select_receding_depth_lines(
            gemini_client,
            sampled_frame,
            args.prompt_for_video,
            candidate_result,
            candidate_overlay_path,
        )
        for sampled_frame, candidate_result, candidate_overlay_path in zip(
            sampled_frames,
            edge_candidate_results,
            edge_candidate_overlay_paths,
            strict=True,
        )
    ]
    line_results = [
        line_result_from_edge_candidate_selection(
            selection["frame_index"],
            candidate_result,
            selection,
        )
        for selection, candidate_result in zip(gemini_line_selections, edge_candidate_results, strict=True)
    ]
    global_vp_estimate = estimate_video_global_vanishing_point(
        line_results,
        width=sampled_frames[0].width,
        height=sampled_frames[0].height,
    )

    diagnostics = []
    annotated_frame_paths: list[Path] = []
    for sampled_frame, edge_result, line_result in zip(sampled_frames, edge_results, line_results, strict=True):
        estimate = estimate_frame_vanishing_point(
            line_result,
            width=sampled_frame.width,
            height=sampled_frame.height,
        )
        annotated_path = runtime_config.annotated_frames_dir / f"frame_{sampled_frame.frame_index:06d}_annotated.png"
        create_annotated_frame(sampled_frame, edge_result, line_result, estimate, annotated_path)
        annotated_frame_paths.append(annotated_path)
        diagnostics.append(
            make_frame_diagnostics(
                sampled_frame,
                edge_result,
                line_result,
                estimate,
                annotated_path,
                runtime_config.project_root,
            ).to_dict()
        )

    write_json(
        runtime_config.gemini_ground_parallel_line_selections_path,
        {
            "line_selection_method": (
                "Full-resolution edge-probability tensor line extraction + "
                "Gemini candidate-ID selection of receding/depth-parallel finite scene lines + "
                "per-frame Lu-style spherical VP voting/inlier clustering. "
                "Frontal left-to-right image-plane-parallel lines are explicitly rejected because "
                "they vanish at infinity rather than at the finite VP being tested."
            ),
            "video_level_vanishing_point_summary_not_used_for_frame_overlays": {
                "x": global_vp_estimate.x,
                "y": global_vp_estimate.y,
                "inside_image": global_vp_estimate.inside_image,
                "inlier_count": global_vp_estimate.inlier_count,
                "inlier_ratio": global_vp_estimate.inlier_ratio,
                "residual_median": global_vp_estimate.residual_median,
            },
            "frames": [
                {
                    "frame_index": frame["frame_index"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "gemini_receding_depth_line_selection": selection,
                    "edge_candidate_line_count": len(candidate_result.accepted_segments),
                    "edge_candidate_overlay_path": selection["edge_candidate_overlay_path"],
                    "candidate_receding_depth_line_segments": frame["accepted_line_segments"],
                    "rejected_line_segment_count": len(frame["rejected_line_segments"]),
                }
                for frame, selection, candidate_result in zip(
                    diagnostics,
                    gemini_line_selections,
                    edge_candidate_results,
                    strict=True,
                )
            ],
        },
    )
    write_json(runtime_config.vanishing_point_diagnostics_path, diagnostics)
    create_contact_sheet(annotated_frame_paths, runtime_config.contact_sheet_path)

    evidence = GeminiEvidence(
        video_path=metadata.video_path,
        prompt_for_video=args.prompt_for_video,
        sampled_frame_paths=[frame.image_path for frame in sampled_frames],
        annotated_frame_paths=annotated_frame_paths,
        contact_sheet_path=runtime_config.contact_sheet_path,
        vanishing_point_diagnostics_path=runtime_config.vanishing_point_diagnostics_path,
        runtime_config=runtime_config,
    )

    vanishing_point_consistency = evaluate_vanishing_point(gemini_client, evidence)
    single_light_source_consistency = evaluate_single_light_source(gemini_client, evidence)
    prompt_object_recognizability = evaluate_prompt_object_recognizability(gemini_client, evidence)

    write_results_json(
        metadata,
        runtime_config,
        args.prompt_for_video,
        sampled_frames,
        vanishing_point_consistency,
        single_light_source_consistency,
        prompt_object_recognizability,
    )
    write_benchmark_report(
        metadata,
        runtime_config,
        args.prompt_for_video,
        sampled_frames,
        vanishing_point_consistency,
        single_light_source_consistency,
        prompt_object_recognizability,
    )

    return (
        vanishing_point_consistency,
        single_light_source_consistency,
        prompt_object_recognizability,
        runtime_config.results_json_path,
        runtime_config.benchmark_report_path,
    )


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    args = build_parser().parse_args()
    (
        vanishing_point_consistency,
        single_light_source_consistency,
        prompt_object_recognizability,
        results_json_path,
        benchmark_report_path,
    ) = run_benchmark(args)

    project_root = Path.cwd().resolve()
    print(f"Vanishing point consistency: {vanishing_point_consistency}")
    print(f"Single light source consistency: {single_light_source_consistency}")
    print(f"Prompt-object recognizability: {prompt_object_recognizability}")
    print(f"Results written to: {display_path(results_json_path, project_root)}")
    print(f"Benchmark report written to: {display_path(benchmark_report_path, project_root)}")
