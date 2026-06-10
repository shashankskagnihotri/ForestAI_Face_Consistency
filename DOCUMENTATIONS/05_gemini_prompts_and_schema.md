# Gemini Prompts And Schema

The required model is:

```text
gemini-3.1-pro-preview
```

The benchmark does not silently switch models.

## Binary Schema

Final yes/no checks use:

```json
{
  "type": "STRING",
  "enum": ["yes", "no"]
}
```

The response MIME type is `text/x.enum`. The code validates that the returned value is exactly `yes` or `no`.

## Receding-Depth Candidate Selection Schema

For each sampled frame, Gemini is asked to select only candidate IDs from an edge-backed manifest. The response contains:

```json
{
  "usable": true,
  "frame_assessment": "short visible-evidence summary",
  "selected_candidate_ids": ["L000", "L001"]
}
```

The sanitizer removes unknown or duplicate IDs and maps accepted IDs back to the original tensor-derived line geometry.

## Vanishing-Point Prompt Requirements

The current prompt defines a vanishing point as the image-plane point where projections of mutually parallel 3D lines converge. It instructs Gemini to use only receding/depth-parallel edges, such as hallway, floor-wall, ceiling-wall, rail, baseboard, table, counter, or rigid rectangular-object edges.

The prompt explicitly rejects:

- left-to-right frontal/transverse lines parallel to the camera plane,
- decorative texture, shadows, reflections, object silhouettes, vegetation, curved boundaries, and non-rigid patterns,
- loose clusters where line extensions merely meet near the same area.

The final yes/no prompt says a true finite VP requires visible strict line extensions to meet at a **perfect single common point** in each successful frame. Residuals around or above 1 degree are treated as loose support, not primary evidence.

## Single-Light Prompt Requirements

The compact Gemini response is constrained to the information needed for debugging:

- final `single_light_source_consistency`,
- a list of light sources,
- each source's direction relative to frame center.

If there are multiple light sources, each source gets its own relative direction.

## Prompt-Object Prompt Requirements

The compact Gemini response is constrained to:

- the objects of interest inferred from the text prompt,
- whether each object is visible overall,
- per-frame visibility for those same objects.

The benchmark does not request chain-of-thought.
