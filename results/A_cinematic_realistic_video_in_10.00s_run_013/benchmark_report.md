# Video Realism Benchmark Report

## Input
- Video path: ./sample/prompt_2/A_cinematic_realistic_video_in.mp4
- Video duration: 10.00 seconds
- Prompt used to generate the video: make a video of a person walking, towards the camera in a long hallway, like in a vatican city palace, a long hallway, 2-3 other people walking away or tangenial to the camera.  The main person is holding a flyer like map of the vatican city, exploring the place.  in the last 3 seconds focus on the face of the person. 

## Sampled Frames
- Frame indices: [0, 11, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 167, 176, 192, 208, 224, 226]
- Frame timestamps, seconds: [0.0, 0.458333, 0.666667, 1.333333, 2.0, 2.666667, 3.333333, 4.0, 4.666667, 5.333333, 6.0, 6.666667, 6.958333, 7.333333, 8.0, 8.666667, 9.333333, 9.416667]

## Final Binary Results
- Vanishing point consistency: yes
- Single light source consistency: yes
- Prompt-object recognizability: yes

## Generated Artifacts
- Raw sampled frames: ./results/A_cinematic_realistic_video_in_10.00s_run_013/sampled_frames
- DexiNed edge probability maps: ./results/A_cinematic_realistic_video_in_10.00s_run_013/edge_maps
- Edge-tensor-derived receding/depth-parallel VP line evidence: ./results/A_cinematic_realistic_video_in_10.00s_run_013/gemini_ground_parallel_line_selections.json
- Annotated vanishing-point visualizations: ./results/A_cinematic_realistic_video_in_10.00s_run_013/annotated_frames
- Vanishing-point diagnostics JSON: ./results/A_cinematic_realistic_video_in_10.00s_run_013/vanishing_point_diagnostics.json
- Contact sheet: ./results/A_cinematic_realistic_video_in_10.00s_run_013/contact_sheet_vanishing_point.png
- Gemini request metadata: ./results/A_cinematic_realistic_video_in_10.00s_run_013/gemini_requests
- Compact Gemini structured responses: ./results/A_cinematic_realistic_video_in_10.00s_run_013/gemini_responses
- Final JSON results: ./results/A_cinematic_realistic_video_in_10.00s_run_013/results.json

## Limitations
- The VP line evidence uses full-resolution edge-tensor line extraction plus Gemini candidate-ID selection and Lu-style spherical voting; it still depends on visible straight scene/object edges being present and recognizable.
- The vanishing-point estimator provides evidence and diagnostics; the final yes/no answer is the strict Gemini binary judgment, not a mathematical proof of scene geometry.
- Insufficient visual evidence is handled conservatively by the Gemini prompt and should produce `no`.
- This report contains no chain-of-thought or hidden reasoning.
