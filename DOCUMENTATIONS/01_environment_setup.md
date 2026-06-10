# Environment Setup

Recommended Python version: 3.11.

Create and activate the conda environment:

```bash
conda create -y -n video_realism_benchmark python=3.11
conda activate video_realism_benchmark
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Required Python libraries include PyAV, OpenCV, PyTorch, Kornia, Pillow, NumPy, Hugging Face Hub, and the official Google GenAI SDK.

## Gemini API Key

Export the API key:

```bash
export GEMINI_ForestAI_API_KEY="your_key_here"
```

The CLI argument `--google_cloud_API_key_name` must be the environment variable name. It defaults to `GEMINI_ForestAI_API_KEY`.

## DexiNed Checkpoint

DexiNed BIPED weights are required for edge detection. The first run checks for:

```text
models/DexiNed_BIPED_10.pth
```

If the checkpoint is missing, it is downloaded automatically from Hugging Face:

```text
kornia/dexined / DexiNed_BIPED_10.pth
```

Optional overrides:

```bash
export DEXINED_WEIGHTS_PATH="/absolute/path/DexiNed_BIPED_10.pth"
export DEXINED_HF_REPO_ID="kornia/dexined"
export DEXINED_HF_FILENAME="DexiNed_BIPED_10.pth"
```

The benchmark does not fall back to Canny or another edge detector. If the checkpoint cannot be downloaded or loaded, the run fails with an explicit error.
