#!/bin/bash

#SBATCH --job-name=forestai_prompt2
#SBATCH --partition=gpu-vram-48gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de
#SBATCH --output=slurm/forestai_prompt2_%j.out
#SBATCH --error=slurm/forestai_prompt2_%j.err

set -euo pipefail

cd /ceph/sagnihot/projects/ForestAI_Face_Consistency

source /ceph/sagnihot/miniconda3/etc/profile.d/conda.sh
conda activate video_realism_benchmark

SAMPLE_ID="${SAMPLE_ID:-prompt_2}"
SAMPLE_DIR="sample/${SAMPLE_ID}"
PROMPT_FILE="${SAMPLE_DIR}/prompt.txt"
VIDEO_PATH="$(find "${SAMPLE_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 1)"
REPORT_DIR="debugging/${SAMPLE_ID}"
REPORT_PATH="${REPORT_DIR}/debug_report.md"

mkdir -p "${REPORT_DIR}" slurm

trap 'status=$?; { echo; echo "## Cycle failed $(date -Is)"; echo; echo "- Exit status: ${status}"; echo "- Slurm stderr: slurm/forestai_prompt2_${SLURM_JOB_ID:-local}.err"; } >> "${REPORT_PATH}"; exit ${status}' ERR

PROMPT="$(tr '\n' ' ' < "${PROMPT_FILE}")"

{
  echo
  echo "## Cycle started $(date -Is)"
  echo
  echo "- Job ID: ${SLURM_JOB_ID:-local}"
  echo "- Sample ID: ${SAMPLE_ID}"
  echo "- Video: ${VIDEO_PATH}"
  echo "- Prompt file: ${PROMPT_FILE}"
  echo "- Gemini key environment variable: GEMINI_ForestAI_API_KEY"
  echo "- Implementation under test: full-resolution edge-tensor line candidates plus Gemini candidate-ID selection and Lu-style VP voting; compact Gemini response JSON."
} >> "${REPORT_PATH}"

python -W ignore main.py \
  --video_path "${VIDEO_PATH}" \
  --prompt_for_video "${PROMPT}" \
  --google_cloud_API_key_name GEMINI_ForestAI_API_KEY

python - "${VIDEO_PATH}" <<'PY' >> "${REPORT_PATH}"
import json
import sys
from pathlib import Path

video_path = Path(sys.argv[1]).resolve()
matching = []
for result_dir in Path("results").iterdir():
    metadata_path = result_dir / "metadata.json"
    if not metadata_path.exists():
        continue
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    recorded = metadata.get("video_path", "")
    if recorded.startswith("./"):
        recorded_path = (Path.cwd() / recorded[2:]).resolve()
    else:
        recorded_path = Path(recorded).resolve()
    if recorded_path == video_path:
        matching.append(result_dir)

latest = max(matching, key=lambda p: p.stat().st_mtime)
results = json.loads((latest / "results.json").read_text(encoding="utf-8"))
diagnostics = json.loads((latest / "vanishing_point_diagnostics.json").read_text(encoding="utf-8"))
light = json.loads((latest / "gemini_responses" / "single_light_source_response.json").read_text(encoding="utf-8"))
objects = json.loads((latest / "gemini_responses" / "prompt_object_recognizability_response.json").read_text(encoding="utf-8"))

accepted_counts = [frame["number_candidate_structural_line_segments"] for frame in diagnostics]
inlier_counts = [frame["number_vanishing_point_inliers"] for frame in diagnostics]
consistent_frames = sum(
    1 for frame in diagnostics
    if frame["geometrically_consistent_with_one_dominant_vanishing_point"]
)

print(f"- Result directory: {latest}")
print(f"- Vanishing point consistency: {results['vanishing_point_consistency']}")
print(f"- Single light-source consistency: {results['single_light_source_consistency']}")
print(f"- Prompt-object recognizability: {results['prompt_object_recognizability']}")
print(f"- Sampled frames: {len(results['sampled_frame_indices'])}")
print(f"- Candidate line counts: {accepted_counts}")
print(f"- VP inlier counts: {inlier_counts}")
print(f"- Geometrically consistent VP frames before Gemini judgment: {consistent_frames}/{len(diagnostics)}")
print(f"- Light-source response: {json.dumps(light, ensure_ascii=True)}")
print(f"- Prompt-object response: {json.dumps(objects, ensure_ascii=True)}")
print(f"- Raw frames: {results['sampled_frames_dir']}")
print(f"- Annotated VP overlays: {results['annotated_frames_dir']}")
print(f"- Contact sheet: {results['contact_sheet_path']}")
print(f"- Gemini responses: {results['gemini_responses_dir']}")
print()
print("### What Worked")
print("- The benchmark completed end to end for the selected prompt sample.")
print("- Gemini response files are compact and contain only validated decision/analysis fields.")
print("- VP diagnostics include edge-tensor-derived receding/depth lines, inlier counts, residuals, overlays, and contact sheet.")
print()
print("### What Needs Review")
if consistent_frames < max(1, len(diagnostics) // 2):
    print("- Too few frames have a geometrically consistent VP; inspect the contact sheet and tune Gemini depth-line selection, line sanitization, or VP clustering.")
else:
    print("- VP consistency is supported in most sampled frames; inspect the contact sheet for any frontal/transverse or non-depth line outliers.")
if not light.get("light_sources"):
    print("- Gemini did not identify any large light-source direction; the light-source prompt/schema may need another pass.")
if not objects.get("objects_of_interest"):
    print("- Gemini did not return prompted objects of interest; the object-visibility prompt/schema may need another pass.")
print()
PY
