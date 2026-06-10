# ForestAI Face Consistency / Video Realism Benchmark

This project evaluates generated videos with deterministic computer-vision evidence and strict Gemini judgments. It is built around one practical question: does a generated video remain physically and semantically consistent with the prompt that created it?

The benchmark currently checks three criteria:

1. **Vanishing point consistency**: real edge-backed receding scene lines should converge to one precise finite vanishing point in each usable frame.
2. **Single light source consistency**: shadows, highlights, shading, and illumination should be compatible with one dominant light source.
3. **Prompt-object recognizability**: objects of interest implied by the text prompt should be visible and recognizable.

Each final criterion returns exactly `yes` or `no`. Detailed Gemini JSON responses are saved for debugging, but final decisions are schema-validated.

## How The Vanishing-Point Check Works

The current vanishing-point pipeline does not let Gemini invent line coordinates. It works like this:

1. Sample frames deterministically from the input video.
2. Run DexiNed on each sampled frame and save the full-resolution edge probability tensor.
3. Extract finite straight-line candidates from that tensor using OpenCV LSD plus probabilistic Hough over thresholded edge probabilities.
4. Filter candidates by edge support, edge confidence, line length, vertical/depth change, duplicate geometry, and central-occluder coverage.
5. Send Gemini the raw frame, an indexed candidate overlay, and a JSON manifest of candidate IDs.
6. Require Gemini to select only candidate IDs that are real receding/depth-parallel scene or object edges.
7. Estimate per-frame vanishing points from those selected real line segments using robust geometric voting and residual checks.
8. Draw only strict inlier evidence in the final annotated frames. Thin extensions are collinear with the detected finite edge segments.

The final Gemini prompt explicitly says a true finite vanishing point requires the accepted line extensions to meet at a **perfect single common point**. Lines that merely pass near each other or form a loose convergence zone are not enough.

## Environment

Create and activate a clean conda environment:

```bash
conda create -y -n video_realism_benchmark python=3.11
conda activate video_realism_benchmark
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies are PyTorch, Kornia, OpenCV, PyAV, Pillow, NumPy, Hugging Face Hub, and the official Google GenAI SDK.

## Gemini API Key

Set the Gemini API key in an environment variable. The CLI takes the environment variable name, not the key itself.

```bash
export GEMINI_ForestAI_API_KEY="your_key_here"
```

The default CLI argument is `--google_cloud_API_key_name GEMINI_ForestAI_API_KEY`, so no extra API-key argument is needed when that variable is set.

## Model Checkpoint

DexiNed BIPED weights are required for edge detection. On first run, if the checkpoint is missing, the code downloads it automatically from Hugging Face:

```text
repo: kornia/dexined
file: DexiNed_BIPED_10.pth
target: models/DexiNed_BIPED_10.pth
```

Optional overrides:

```bash
export DEXINED_WEIGHTS_PATH="/absolute/path/DexiNed_BIPED_10.pth"
export DEXINED_HF_REPO_ID="kornia/dexined"
export DEXINED_HF_FILENAME="DexiNed_BIPED_10.pth"
```

`models/` is ignored by git because checkpoints are large runtime artifacts.

## Run

Use `python -W ignore` to keep dependency warnings out of long benchmark logs.

For the current prompt_2 debugging sample:

```bash
PROMPT="$(cat sample/prompt_2/prompt.txt)"
python -W ignore main.py \
  --video_path sample/prompt_2/A_cinematic_realistic_video_in.mp4 \
  --prompt_for_video "$PROMPT"
```

General form:

```bash
python -W ignore main.py \
  --video_path /path/to/video.mp4 \
  --prompt_for_video "the original video-generation prompt"
```

CLI arguments:

- `--video_path`: required path to the `.mp4` video.
- `--prompt_for_video`: required text prompt used to generate the video.
- `--google_cloud_API_key_name`: optional environment-variable name containing the Gemini API key. Defaults to `GEMINI_ForestAI_API_KEY`.

## Slurm

A sample Slurm launcher is provided:

```bash
sbatch scripts/run_sample.sbatch
```

It is configured for the sample workflow and writes scheduler logs under `slurm/`.

## Outputs

Benchmark outputs are written under:

```text
results/{safe_video_name}_{duration_seconds}s/
```

If a result directory already exists, the benchmark appends `_run_001`, `_run_002`, and so on.

Important artifacts:

- `results.json`: final validated `yes`/`no` decisions and artifact paths.
- `benchmark_report.md`: concise human-readable benchmark report.
- `metadata.json`: video metadata, sampling rule, and resolved paths.
- `sampled_frames/`: sampled raw frames.
- `edge_maps/`: DexiNed edge PNGs and exact `.npy` probability tensors.
- `annotated_frames/`: candidate overlays and strict VP evidence frames.
- `contact_sheet_vanishing_point.png`: visual summary of annotated frames.
- `gemini_requests/`: prompt, schema, evidence paths, and candidate manifests. API keys are never written.
- `gemini_responses/`: exact Gemini responses and validated decisions.
- `vanishing_point_diagnostics.json`: detailed per-frame geometry and residuals.

Development reports are written under `debugging/`. Runtime outputs, debugging artifacts, Slurm logs, and model checkpoints are ignored by git.

## Documentation

Detailed docs live in `DOCUMENTATIONS/`:

- `00_overview.md`
- `01_environment_setup.md`
- `02_running_the_benchmark.md`
- `03_file_by_file_documentation.md`
- `04_methodology.md`
- `05_gemini_prompts_and_schema.md`
- `06_outputs_and_reports.md`

## License

This project is licensed under the GNU General Public License v3.0. See `LICENSE`.
