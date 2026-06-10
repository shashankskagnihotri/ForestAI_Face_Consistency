# Running The Benchmark

Use `python -W ignore` for clean logs:

```bash
PROMPT="$(cat sample/prompt_2/prompt.txt)"
python -W ignore main.py \
  --video_path sample/prompt_2/A_cinematic_realistic_video_in.mp4 \
  --prompt_for_video "$PROMPT"
```

General command:

```bash
python -W ignore main.py \
  --video_path /path/to/video.mp4 \
  --prompt_for_video "the original video-generation prompt"
```

Arguments:

- `--video_path`: required video path.
- `--prompt_for_video`: required prompt used to generate the video.
- `--google_cloud_API_key_name`: optional name of the environment variable containing the Gemini API key. Defaults to `GEMINI_ForestAI_API_KEY`.

The benchmark validates the video, samples frames, runs DexiNed edge detection, extracts edge-backed line candidates, asks Gemini to select receding/depth-parallel candidate IDs, estimates vanishing points, creates visual evidence, runs the three Gemini checks, and writes outputs under `results/`.

Expected terminal output:

```text
Vanishing point consistency: yes/no
Single light source consistency: yes/no
Prompt-object recognizability: yes/no
Results written to: ./results/{safe_video_name}_{video_length_seconds}s/results.json
Benchmark report written to: ./results/{safe_video_name}_{video_length_seconds}s/benchmark_report.md
```

## Slurm

For cluster runs, use:

```bash
sbatch scripts/run_sample.sbatch
```

Monitor with:

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
```
