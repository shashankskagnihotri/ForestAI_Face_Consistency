# Video Realism Benchmark Report

## Input
- Video path: ./sample/prompt_1/make_a_seconds_video_of_a_m.mp4
- Video duration: 10.00 seconds
- Prompt used to generate the video: make a 15 seconds video of a man sitting at a table in a resturant eating a vegan burger, zoom into the face of the man in the last 3 seconds of the video.

## Sampled Frames
- Frame indices: [0, 11, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 167, 176, 192, 208, 224, 226]
- Frame timestamps, seconds: [0.0, 0.458333, 0.666667, 1.333333, 2.0, 2.666667, 3.333333, 4.0, 4.666667, 5.333333, 6.0, 6.666667, 6.958333, 7.333333, 8.0, 8.666667, 9.333333, 9.416667]

## Final Binary Results
- Vanishing point consistency: no
- Single light source consistency: yes
- Prompt-object recognizability: yes

## Generated Artifacts
- Raw sampled frames: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/sampled_frames
- DexiNed edge probability maps: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/edge_maps
- Edge-tensor-derived receding/depth-parallel VP line evidence: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/gemini_ground_parallel_line_selections.json
- Annotated vanishing-point visualizations: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/annotated_frames
- Vanishing-point diagnostics JSON: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/vanishing_point_diagnostics.json
- Contact sheet: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/contact_sheet_vanishing_point.png
- Gemini request metadata: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/gemini_requests
- Compact Gemini structured responses: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/gemini_responses
- Final JSON results: ./results/make_a_seconds_video_of_a_m_10.00s_run_007/results.json

## Limitations
- The VP line evidence uses full-resolution edge-tensor line extraction plus Gemini candidate-ID selection and Lu-style spherical voting; it still depends on visible straight scene/object edges being present and recognizable.
- The vanishing-point estimator provides evidence and diagnostics; the final yes/no answer is the strict Gemini binary judgment, not a mathematical proof of scene geometry.
- Insufficient visual evidence is handled conservatively by the Gemini prompt and should produce `no`.
- This report contains no chain-of-thought or hidden reasoning.
