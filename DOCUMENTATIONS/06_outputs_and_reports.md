# Outputs And Reports

Actual benchmark results are stored in:

```text
results/{safe_video_name}_{video_length_seconds}s/
```

If that directory already exists, deterministic suffixes are appended:

```text
results/{safe_video_name}_{video_length_seconds}s_run_001/
results/{safe_video_name}_{video_length_seconds}s_run_002/
```

Development reports are stored under:

```text
debugging/
```

`debugging/` is not part of the benchmark output and should not be used for sampled frames, edge maps, annotations, Gemini responses, `results.json`, `benchmark_report.md`, or `metadata.json`.

## Per-Video Result Directory

- `results.json`: final machine-readable result with validated decisions and artifact paths.
- `benchmark_report.md`: human-readable benchmark result report without chain-of-thought.
- `metadata.json`: video metadata, sampling rule, random seed, and resolved result directory.
- `vanishing_point_diagnostics.json`: per-frame candidate, inlier, residual, and VP-coordinate diagnostics.
- `vanishing_point_compact_diagnostics.json`: compact VP diagnostics passed into final judging.
- `gemini_ground_parallel_line_selections.json`: edge-candidate selection method and selected candidate evidence.
- `contact_sheet_vanishing_point.png`: compact visual summary of strict annotated VP frames.
- `sampled_frames/`: raw sampled frame PNGs.
- `edge_maps/`: DexiNed edge PNGs and exact probability `.npy` files.
- `annotated_frames/`: candidate-ID overlays and strict VP annotated frames.
- `gemini_requests/`: request metadata, prompts, schemas, candidate manifests, and evidence paths. API keys are never stored.
- `gemini_responses/`: exact Gemini structured responses and validated final decisions.
