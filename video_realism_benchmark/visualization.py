"""Annotated frame and contact-sheet generation for Gemini evidence."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import config
from .edge_detection import EdgeMapResult
from .frame_sampling import SampledFrame
from .line_detection import LineDetectionResult, LineSegment
from .line_detection import line_candidate_id
from .vanishing_point import VanishingPointEstimate


COLOR_REJECTED = (60, 60, 220)
COLOR_ACCEPTED_OUTLIER = (0, 190, 255)
COLOR_ACCEPTED_INLIER = (40, 220, 80)
COLOR_EXTENDED = (255, 210, 0)
COLOR_VP = (255, 0, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_CANDIDATES: tuple[tuple[int, int, int], ...] = (
    (40, 220, 80),
    (0, 190, 255),
    (255, 210, 0),
    (255, 120, 80),
    (255, 0, 255),
    (120, 220, 255),
)


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """Draw readable text with a solid background."""

    x, y = origin
    size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.VIS_FONT_SCALE,
        config.VIS_FONT_THICKNESS,
    )
    cv2.rectangle(
        image,
        (x - 4, y - size[1] - baseline - 4),
        (x + size[0] + 4, y + baseline + 4),
        COLOR_TEXT_BG,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.VIS_FONT_SCALE,
        COLOR_TEXT,
        config.VIS_FONT_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def _draw_segment(image: np.ndarray, segment: LineSegment, color: tuple[int, int, int], thickness: int) -> None:
    """Draw one finite detected line segment."""

    cv2.line(
        image,
        (int(round(segment.x1)), int(round(segment.y1))),
        (int(round(segment.x2)), int(round(segment.y2))),
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _extended_line_endpoints(segment: LineSegment, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Intersect the infinite segment line with the image rectangle."""

    x1, y1, x2, y2 = segment.x1, segment.y1, segment.x2, segment.y2
    dx = x2 - x1
    dy = y2 - y1
    points: list[tuple[float, float]] = []

    if abs(dx) > 1e-9:
        for x in (0.0, float(width - 1)):
            t = (x - x1) / dx
            y = y1 + t * dy
            if -1e-6 <= y <= height - 1 + 1e-6:
                points.append((x, y))
    if abs(dy) > 1e-9:
        for y in (0.0, float(height - 1)):
            t = (y - y1) / dy
            x = x1 + t * dx
            if -1e-6 <= x <= width - 1 + 1e-6:
                points.append((x, y))

    unique: list[tuple[float, float]] = []
    for point in points:
        if not any(math.hypot(point[0] - other[0], point[1] - other[1]) < 1.0 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None

    best_pair = max(
        ((a, b) for index, a in enumerate(unique) for b in unique[index + 1 :]),
        key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]),
    )
    return (
        (int(round(best_pair[0][0])), int(round(best_pair[0][1]))),
        (int(round(best_pair[1][0])), int(round(best_pair[1][1]))),
    )


def _draw_vanishing_point(image: np.ndarray, estimate: VanishingPointEstimate) -> None:
    """Draw the VP if inside the image, otherwise annotate its off-frame location."""

    if estimate.x is None or estimate.y is None:
        _put_label(image, "VP: not estimated", (10, image.shape[0] - 12))
        return
    height, width = image.shape[:2]
    x = float(estimate.x)
    y = float(estimate.y)
    if 0 <= x <= width - 1 and 0 <= y <= height - 1:
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, 8, COLOR_VP, thickness=2, lineType=cv2.LINE_AA)
        _put_label(image, f"VP ({x:.1f}, {y:.1f})", (max(10, center[0] + 10), max(24, center[1] - 8)))
    else:
        _put_label(image, f"VP outside ({x:.1f}, {y:.1f})", (10, image.shape[0] - 12))


def _is_strict_visual_inlier(segment: LineSegment) -> bool:
    """Return whether a segment is tight enough to visualize as VP evidence."""

    return (
        bool(segment.vp_inlier)
        and segment.vp_residual_deg is not None
        and float(segment.vp_residual_deg) <= config.VP_VISUAL_STRICT_INLIER_THRESHOLD_DEG
    )


def _draw_segment_aligned_extension_toward_vp(
    image: np.ndarray,
    segment: LineSegment,
    estimate: VanishingPointEstimate,
    color: tuple[int, int, int],
) -> None:
    """Draw a VP-facing extension that stays collinear with the finite edge segment."""

    if estimate.x is None or estimate.y is None:
        return
    vp = np.array([float(estimate.x), float(estimate.y)], dtype=np.float64)
    height, width = image.shape[:2]
    if not (0.0 <= vp[0] <= width - 1 and 0.0 <= vp[1] <= height - 1):
        return

    p1 = np.array([segment.x1, segment.y1], dtype=np.float64)
    p2 = np.array([segment.x2, segment.y2], dtype=np.float64)
    direction = p2 - p1
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    direction /= norm

    t1 = float(np.dot(p1, direction))
    t2 = float(np.dot(p2, direction))
    t_vp = float(np.dot(vp, direction))
    if t_vp <= min(t1, t2):
        start = p1 if t1 <= t2 else p2
    elif t_vp >= max(t1, t2):
        start = p1 if t1 >= t2 else p2
    else:
        start = min((p1, p2), key=lambda point: float(np.linalg.norm(point - vp)))

    target = p1 + np.dot(vp - p1, direction) * direction
    if not (-1.0 <= target[0] <= width and -1.0 <= target[1] <= height):
        endpoints = _extended_line_endpoints(segment, width, height)
        if endpoints is None:
            return
        candidates = [np.array(point, dtype=np.float64) for point in endpoints]
        target = min(candidates, key=lambda point: float(np.linalg.norm(point - vp)))
    if float(np.linalg.norm(target - start)) < 3.0:
        return

    cv2.line(
        image,
        (int(round(start[0])), int(round(start[1]))),
        (int(round(target[0])), int(round(target[1]))),
        color,
        thickness=config.VIS_EXTENDED_LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )


def create_annotated_frame(
    sampled_frame: SampledFrame,
    edge_result: EdgeMapResult,
    line_result: LineDetectionResult,
    estimate: VanishingPointEstimate,
    output_path: Path,
) -> Path:
    """Create a three-panel visualization for one sampled frame."""

    bgr = cv2.imread(str(sampled_frame.image_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise RuntimeError(f"Could not load sampled frame for visualization: {sampled_frame.image_path}")
    height, width = bgr.shape[:2]

    original_panel = bgr.copy()
    _put_label(
        original_panel,
        f"original frame {sampled_frame.frame_index} t={sampled_frame.timestamp_sec:.3f}s",
        (10, 24),
    )

    edge_uint8 = np.round(edge_result.probability_map * 255.0).astype(np.uint8)
    edge_panel = cv2.cvtColor(edge_uint8, cv2.COLOR_GRAY2BGR)
    _put_label(edge_panel, "DexiNed edge probability", (10, 24))

    annotated = bgr.copy()
    extension_layer = annotated.copy()
    for segment in line_result.accepted_segments:
        if _is_strict_visual_inlier(segment):
            _draw_segment_aligned_extension_toward_vp(extension_layer, segment, estimate, COLOR_EXTENDED)
    annotated = cv2.addWeighted(extension_layer, 0.55, annotated, 0.45, 0.0)

    for segment in line_result.accepted_segments:
        if _is_strict_visual_inlier(segment):
            _draw_segment(annotated, segment, COLOR_ACCEPTED_INLIER, config.VIS_LINE_THICKNESS)

    _draw_vanishing_point(annotated, estimate)
    _put_label(
        annotated,
        "strict edge-tensor depth lines; thin extensions follow each line",
        (10, 24),
    )

    composite = np.concatenate([original_panel, edge_panel, annotated], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), composite):
        raise RuntimeError(f"Failed to write annotated frame: {output_path}")
    return output_path


def create_edge_candidate_selection_frame(
    sampled_frame: SampledFrame,
    edge_result: EdgeMapResult,
    candidate_result: LineDetectionResult,
    output_path: Path,
) -> Path:
    """Create an indexed visualization of real edge-tensor line candidates."""

    bgr = cv2.imread(str(sampled_frame.image_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise RuntimeError(f"Could not load sampled frame for candidate visualization: {sampled_frame.image_path}")

    original_panel = bgr.copy()
    _put_label(
        original_panel,
        f"original frame {sampled_frame.frame_index} t={sampled_frame.timestamp_sec:.3f}s",
        (10, 24),
    )

    edge_uint8 = np.round(edge_result.probability_map * 255.0).astype(np.uint8)
    edge_panel = cv2.cvtColor(edge_uint8, cv2.COLOR_GRAY2BGR)
    _put_label(edge_panel, "edge tensor values", (10, 24))

    candidate_panel = bgr.copy()
    for index, segment in enumerate(candidate_result.accepted_segments):
        color = COLOR_CANDIDATES[index % len(COLOR_CANDIDATES)]
        _draw_segment(candidate_panel, segment, color, config.VIS_LINE_THICKNESS)
        label_x = int(round(segment.midpoint_x))
        label_y = int(round(segment.midpoint_y))
        _put_label(candidate_panel, line_candidate_id(segment), (max(8, label_x), max(24, label_y)))

    _put_label(
        candidate_panel,
        "edge-tensor candidate IDs only",
        (10, 24),
    )

    composite = np.concatenate([original_panel, edge_panel, candidate_panel], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), composite):
        raise RuntimeError(f"Failed to write edge candidate frame: {output_path}")
    return output_path


def create_contact_sheet(annotated_frame_paths: list[Path], output_path: Path) -> Path:
    """Create a compact contact sheet from annotated frame visualizations."""

    if not annotated_frame_paths:
        raise RuntimeError("Cannot create a contact sheet with no annotated frames.")

    images: list[Image.Image] = []
    for path in annotated_frame_paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            scale = config.VIS_CONTACT_SHEET_THUMB_WIDTH / float(rgb.width)
            thumb_height = max(1, int(round(rgb.height * scale)))
            images.append(rgb.resize((config.VIS_CONTACT_SHEET_THUMB_WIDTH, thumb_height), Image.Resampling.LANCZOS))

    columns = min(config.VIS_CONTACT_SHEET_MAX_COLUMNS, max(1, math.ceil(math.sqrt(len(images)))))
    rows = math.ceil(len(images) / columns)
    cell_width = config.VIS_CONTACT_SHEET_THUMB_WIDTH
    cell_height = max(image.height for image in images)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        x = col * cell_width
        y = row * cell_height
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + cell_width - 1, y + image.height - 1), outline=(230, 230, 230), width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path
