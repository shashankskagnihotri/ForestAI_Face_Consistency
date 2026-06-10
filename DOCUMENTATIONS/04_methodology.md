# Methodology

## Frame Sampling

The benchmark samples every 16th frame and additionally samples 3 uniformly random frames. The random seed is fixed in `video_realism_benchmark/config.py`. Sampled frame indices are deduplicated and sorted.

PyAV validates and decodes the video. The benchmark fails if the video path does not exist, no video stream is available, frames cannot decode, dimensions are invalid, FPS is invalid, or timestamps cannot be validated.

## Edge Detection

The edge detector is DexiNed through Kornia with BIPED weights. The model is loaded once and run in deterministic inference mode for every sampled frame.

The code saves:

- edge PNGs for visual inspection,
- full-resolution `.npy` edge probability tensors for line extraction.

No Canny fallback is used. Missing default weights are downloaded from Hugging Face on first run.

## Edge-Backed Line Candidate Extraction

Vanishing-point candidates are extracted from the edge probability tensor, not drawn manually. The pipeline uses:

- OpenCV LSD on the normalized DexiNed probability image,
- probabilistic Hough on the thresholded DexiNed tensor,
- edge-confidence and edge-support checks sampled along each finite segment,
- length and duplicate filters,
- rejection of near-horizontal/frontal lines that are parallel to the image plane,
- rejection of candidates dominated by the central moving subject/occluder.

Candidate overlays show numbered candidate IDs. Gemini sees the raw frame, candidate overlay, and a JSON manifest of candidate geometry, then returns only selected candidate IDs.

## Vanishing-Point Estimation

Selected candidate IDs are converted back into the real finite line segments extracted from the edge tensor. Pairwise intersections and Lu-style spherical voting produce vanishing-point hypotheses. A per-frame robust estimator marks inliers by angular residual.

Final annotated frames show only strict VP inlier evidence, with thin collinear extensions that follow the detected segment geometry. This avoids drawing artificial endpoint-to-VP helper rays.

## Gemini Judging

Gemini receives compact evidence and schema constraints:

- Vanishing point: final answer `yes` only when strict receding/depth-parallel line extensions meet at a perfect single common point in usable frames.
- Single light source: compact JSON listing the dominant source direction relative to frame center, or multiple source directions if present.
- Prompt-object recognizability: compact JSON listing prompt-derived objects of interest, overall visibility, and per-frame visibility.

The final binary judgments are validated to exactly `yes` or `no`.
