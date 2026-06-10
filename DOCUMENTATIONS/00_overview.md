# Overview

`video_realism_benchmark` evaluates generated videos using deterministic visual evidence and strict Gemini judgments.

The benchmark evaluates exactly three criteria:

1. **Vanishing point consistency**: receding/depth-parallel straight scene or rigid-object edges should converge to one precise finite vanishing point in each usable frame.
2. **Single light source consistency**: shadows, highlights, shading gradients, cast-shadow directions, self-shadows, and illumination falloff should match one dominant light source.
3. **Prompt-object recognizability**: the objects of interest implied by the original generation prompt should be visible and recognizable overall and per sampled frame.

The final result for each criterion is exactly `yes` or `no`.

The computer-vision pipeline produces evidence. Gemini makes the final schema-validated judgments from that evidence. For vanishing-point analysis, Gemini selects only real edge-candidate IDs; it does not draw lines or invent coordinates.
