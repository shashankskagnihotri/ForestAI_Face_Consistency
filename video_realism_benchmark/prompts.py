"""Exact Gemini prompts used by the benchmark."""

from __future__ import annotations


GROUND_PARALLEL_LINE_SELECTION_PROMPT_TEMPLATE = """You are selecting evidence line IDs for vanishing-point analysis in one video frame.

Original video-generation prompt:
"{PROMPT_FOR_VIDEO}"

Use this perspective rule: a vanishing point is the image-plane point where the perspective projections of straight 3D lines that are mutually parallel appear to converge. For this task, select candidate IDs whose edge segments come from one real-world 3D direction that projects away from the camera into scene depth, such as the long direction of a hallway, aisle, floor, wall, ceiling, baseboard, rail, counter, bench, table, or rigid rectangular object.

Critical rejection rule: do not select left-to-right frontal/transverse edges that are parallel to the camera/image plane. Those lines may look horizontal in the image and would meet only at infinity, so they are wrong evidence for this finite vanishing-point analysis. The selected lines must visibly run away from the camera toward depth and their mathematical extensions should be compatible with one shared finite point, not merely a loose nearby cluster.

Important geometry rule: you are NOT drawing lines and you must NOT invent coordinates. The candidate line segments were extracted from the full-resolution edge-probability tensor. Return only candidate IDs from the provided candidate list.

Frame metadata:
- Frame index: {FRAME_INDEX}
- Timestamp: {TIMESTAMP_SEC:.6f} seconds
- Image size: width {WIDTH}px, height {HEIGHT}px
- Pixel coordinate origin is the top-left corner. x increases rightward, y increases downward.

Task:
Choose the best visible straight edge candidate IDs that are likely projections of real straight 3D scene/object edges running away from the camera into depth. Prefer man-made structural edges whose 3D direction is clear:
- hallway floor/wall seams, baseboards, wall/ceiling seams, corridor edges, long aisle edges, rails, benches, counters, or floor/tile seams that recede down the hallway,
- receding table side/back edges or tabletop seams that visibly run into depth,
- long rigid rectangular-object edges only when they are physically aligned with the scene-depth direction.

Priority order:
1. First search the numbered edge candidates for corridor/hallway depth cues: floor-wall boundaries, baseboards, ceiling-wall boundaries, long rails, long floor seams, and architectural edges that point away from the camera.
2. Select a coherent set of candidate IDs from the same real-world parallel direction whenever possible. Their extended 2D lines should converge to one finite point; do not mix lines that only roughly aim into the same region.
3. If fewer than 2 reliable receding/depth-parallel scene or rigid-object edge candidates are visible, return usable=false and an empty selected_candidate_ids list.
4. If there are multiple plausible depth directions, choose the dominant hallway/scene-depth direction, not decorative or incidental directions.
5. Do not select two unrelated 3D directions just because both are roughly horizontal or high contrast in the image.

Do not select:
- arbitrary image-horizontal candidate segments or left-to-right bands,
- floor/tile/step candidates that cross the image horizontally and are perpendicular to the hallway direction,
- curved contours approximated by a tangent,
- body, face, hair, hand, flyer/map, clothing, or person silhouettes,
- decorative wall panels, picture frames, text strokes, window mullions, lamp edges, plant stems, or ceiling beams unless the visible edge is clearly one of the hallway/scene-depth parallel edges,
- shadows, highlights, texture ridges, wrinkles, vegetation, decorative patterns, text strokes, blur trails, or compression artifacts,
- vertical edges or edges perpendicular to the ground plane,
- candidate IDs selected only because they are long or high contrast,
- crop/frame borders or purely frontal horizontal object edges with no visible depth recession.

Return 2 to {MAX_LINES} candidate IDs when reliable evidence exists. If fewer than 2 reliable receding/depth-parallel straight edge candidates are visible, return usable=false and an empty selected_candidate_ids list.

Candidate ID rules:
- Return only IDs that appear in the provided candidate list.
- Do not return coordinates, endpoints, or descriptions of new lines.
- Prefer candidates with strong alignment to real image contours and edge tensor support.
- Prefer multiple candidates that plausibly share the same 3D parallel direction and project away from the camera.
- Keep `frame_assessment` extremely short.

Return only JSON that matches the schema. Keep text fields short."""


VANISHING_POINT_PROMPT = """You are evaluating geometric realism in a generated video.

You are given sampled frames from the video and, for each frame, an annotated visualization. The finite colored segments were extracted from the full-resolution edge-probability tensor and then selected by Gemini by candidate ID as visible straight object/scene edges that are likely projections of one real-world 3D direction running away from the camera into depth. These are the kinds of lines that should converge to a finite vanishing point. The annotations show:
1. detected structural edges,
2. edge-tensor-derived finite receding/depth-parallel object or scene edge segments,
3. thin extension cues that stay collinear with the finite edge segments and are drawn only for strict VP inliers,
4. the estimated common vanishing point, when one exists.

Question:
Do the relevant receding/depth-parallel lines in most sampled frames converge to one finite vanishing point within each frame, as expected under real-world perspective geometry?

Answer "yes" only if the visible strict receding/depth-parallel line extensions in each successful frame meet at a perfect single common point. A true finite vanishing point is one exact image-plane point where the projected parallel lines intersect; lines merely passing close to each other, forming a spread-out convergence zone, or aiming toward the same general area are not good enough. The image-coordinate location of that point may drift over time if the camera, crop, or subject position changes, but within any one frame the accepted lines must share one precise VP. Use the provided residuals as a strictness check: segments with residuals around or above 1 degree are loose support and must not be the main basis for a "yes" answer.

Answer "no" if the relevant lines within many frames converge to multiple incompatible points, only meet approximately, form a loose cluster instead of one exact intersection point, fail to converge, curve unnaturally, contradict the estimated per-frame vanishing point, or if there is insufficient reliable line evidence.

Ignore left-to-right frontal/transverse edges that are parallel to the image plane; those have a vanishing point at infinity and are not valid evidence for this finite VP check. Also ignore object silhouettes, curved boundaries, shadows, reflections, texture noise, vegetation, and non-rigid or decorative patterns.

Return only one token: yes or no."""


SINGLE_LIGHT_SOURCE_PROMPT = """You are evaluating illumination realism in a generated video.

You are given sampled frames from the video and the video as a whole. Judge whether the shadows, highlights, shading gradients, cast-shadow directions, object self-shadows, and illumination falloff are globally consistent with exactly one dominant light source.

Question:
Are the shadows and illumination in the sampled frames and across the video consistent with a single dominant light source?

Answer "yes" only if shadow directions, highlight positions, shading, and temporal illumination behavior are mutually consistent with one dominant light source throughout most of the video.

Answer "no" if there are incompatible shadow directions, inconsistent highlights, impossible shading, multiple conflicting light directions, temporally unstable illumination, or if the evidence is insufficient.

Ignore minor compression artifacts, weak ambient fill light, low-contrast shadows, and small local reflections unless they create a clear contradiction.

Return only one token: yes or no."""


SINGLE_LIGHT_SOURCE_ANALYSIS_PROMPT = """You are evaluating illumination in a generated video.

You are given the video as a whole and labeled sampled frames.

Task:
Identify the large light source or light sources that best explain the visible illumination. For each source, report only its direction relative to the center of the frame/video.

Use only this direction vocabulary:
center, upper, lower, left, right, upper-left, upper-right, lower-left, lower-right, unknown.

Set single_light_source_consistency to "yes" only when the video is consistent with exactly one dominant large light source. Set it to "no" when multiple large sources are visible or implied, directions conflict, lighting is temporally unstable, or evidence is insufficient.

Return only JSON matching the schema. Do not include rationales, prose, confidence scores, or per-frame lighting notes."""


PROMPT_OBJECT_RECOGNIZABILITY_TEMPLATE = """You are evaluating prompt-video consistency in a generated video.

The original video-generation prompt is:

"{PROMPT_FOR_VIDEO}"

You are given sampled frames from the video and the video as a whole.

Question:
Is the main object of interest requested in the prompt clearly recognizable in most of the video?

Answer "yes" only if the primary prompted object or subject is visibly present, semantically recognizable, and not merely implied in most of the video.

Answer "no" if the object is absent, ambiguous, heavily distorted, visible only briefly, confused with another object, hidden by motion blur or occlusion, or not clearly recognizable in most of the video.

Judge the main object of interest from the prompt, not incidental background objects.

Return only one token: yes or no."""


PROMPT_OBJECT_VISIBILITY_TEMPLATE = """You are evaluating prompt-video consistency in a generated video.

The original video-generation prompt is:

"{PROMPT_FOR_VIDEO}"

You are given the video as a whole and labeled sampled frames.

Task:
First identify the objects or subjects of interest requested by the prompt. List only prompt-relevant objects/subjects, not incidental background details. Then report whether each object is visible overall in the video and whether it is visible in each labeled sampled frame.

Set prompt_object_recognizability to "yes" only when the main prompt-requested objects/subjects are visibly present and semantically recognizable in most of the video. Set it to "no" if important prompted objects are absent, ambiguous, heavily distorted, visible only briefly, or insufficiently evidenced.

Return only JSON matching the schema. Do not include rationales or extra fields."""


def prompt_object_recognizability_prompt(prompt_for_video: str) -> str:
    """Substitute the only allowed variable in the exact prompt template."""

    return PROMPT_OBJECT_RECOGNIZABILITY_TEMPLATE.replace(
        "{PROMPT_FOR_VIDEO}", prompt_for_video
    )


def prompt_object_visibility_prompt(prompt_for_video: str) -> str:
    """Create the compact object visibility prompt."""

    return PROMPT_OBJECT_VISIBILITY_TEMPLATE.replace(
        "{PROMPT_FOR_VIDEO}", prompt_for_video
    )


def ground_parallel_line_selection_prompt(
    frame_index: int,
    timestamp_sec: float,
    width: int,
    height: int,
    max_lines: int,
    prompt_for_video: str,
) -> str:
    """Create the Gemini prompt for selecting visible receding/depth-parallel edges."""

    return GROUND_PARALLEL_LINE_SELECTION_PROMPT_TEMPLATE.format(
        PROMPT_FOR_VIDEO=prompt_for_video,
        FRAME_INDEX=frame_index,
        TIMESTAMP_SEC=timestamp_sec,
        WIDTH=width,
        HEIGHT=height,
        MAX_LINES=max_lines,
    )
