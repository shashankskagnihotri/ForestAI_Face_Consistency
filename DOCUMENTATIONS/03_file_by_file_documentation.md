# File By File Documentation

## Top Level

- `main.py`: CLI entrypoint that calls `video_realism_benchmark.cli.main`.
- `requirements.txt`: Python dependencies for the conda environment.
- `README.md`: Project purpose, installation, runtime command, outputs, and limitations.

## Package

- `video_realism_benchmark/__init__.py`: package metadata.
- `video_realism_benchmark/cli.py`: argparse CLI and end-to-end orchestration.
- `video_realism_benchmark/config.py`: centralized constants for model name, sampling, DexiNed checkpoint download, edge detection, line filtering, vanishing-point estimation, visualization, and output names.
- `video_realism_benchmark/frame_sampling.py`: deterministic frame index sampling and sampled-frame extraction.
- `video_realism_benchmark/video_io.py`: video validation, PyAV probing, metadata extraction, and timestamp validation.
- `video_realism_benchmark/edge_detection.py`: DexiNed checkpoint resolution/download, model loading, deterministic inference, and edge probability map saving.
- `video_realism_benchmark/line_detection.py`: edge-tensor-backed LSD plus probabilistic Hough segment extraction, candidate-ID payloads, confidence filtering, duplicate removal, and Gemini-selected line reconstruction.
- `video_realism_benchmark/vanishing_point.py`: homogeneous-line conversion, robust intersection/voting, residual computation, inlier marking, and diagnostics serialization.
- `video_realism_benchmark/visualization.py`: edge-candidate overlays, strict VP annotated evidence, and contact sheet generation.
- `video_realism_benchmark/gemini_client.py`: official Google GenAI SDK client, video upload, candidate-ID line selection, compact JSON checks, structured enum outputs, response validation, and request/response metadata saving.
- `video_realism_benchmark/prompts.py`: exact Gemini prompts.
- `video_realism_benchmark/report_writer.py`: `metadata.json`, `results.json`, and `benchmark_report.md` writers.
- `video_realism_benchmark/schemas.py`: strict yes/no schema and validation.
- `video_realism_benchmark/utils.py`: path sanitization, result directory creation, JSON writing, logging setup, and debugging directory policy.

## Documentation And Reports

- `DOCUMENTATIONS/00_overview.md`: benchmark goal and criteria.
- `DOCUMENTATIONS/01_environment_setup.md`: conda, pip, DexiNed weights, and API key setup.
- `DOCUMENTATIONS/02_running_the_benchmark.md`: exact run command and final CLI behavior.
- `DOCUMENTATIONS/03_file_by_file_documentation.md`: this file.
- `DOCUMENTATIONS/04_methodology.md`: deterministic pipeline and Gemini judging method.
- `DOCUMENTATIONS/05_gemini_prompts_and_schema.md`: exact prompts and yes/no schema.
- `DOCUMENTATIONS/06_outputs_and_reports.md`: output directories and report policy.
- `debugging/`: local Codex/debugging reports only. This directory is ignored by git.
