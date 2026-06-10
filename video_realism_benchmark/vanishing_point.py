"""Deterministic robust vanishing-point estimation from structural lines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from pathlib import Path

import numpy as np

from . import config
from .edge_detection import EdgeMapResult
from .frame_sampling import SampledFrame
from .line_detection import LineDetectionResult, LineSegment
from .utils import display_path


@dataclass(frozen=True)
class VanishingPointEstimate:
    """Dominant VP estimate for one sampled frame."""

    x: float | None
    y: float | None
    inside_image: bool | None
    inlier_count: int
    inlier_ratio: float
    residual_mean: float | None
    residual_median: float | None
    residual_max: float | None
    geometrically_consistent: bool


@dataclass(frozen=True)
class FrameVanishingPointDiagnostics:
    """Serializable per-frame vanishing-point diagnostics."""

    frame_index: int
    timestamp_sec: float
    number_detected_line_segments: int
    number_candidate_structural_line_segments: int
    number_vanishing_point_inliers: int
    inlier_ratio: float
    residual_mean: float | None
    residual_median: float | None
    residual_max: float | None
    estimated_vanishing_point_coordinates: list[float] | None
    vanishing_point_inside_image: bool | None
    geometrically_consistent_with_one_dominant_vanishing_point: bool
    raw_sampled_frame_path: str
    edge_map_path: str
    annotated_frame_path: str
    accepted_line_segments: list[dict[str, object]]
    rejected_line_segments: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready diagnostics dictionary."""

        return asdict(self)


def _homogeneous_line(segment: LineSegment) -> np.ndarray:
    """Convert a segment endpoint pair into a normalized homogeneous line."""

    p1 = np.array([segment.x1, segment.y1, 1.0], dtype=np.float64)
    p2 = np.array([segment.x2, segment.y2, 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm <= 0:
        raise RuntimeError(f"Degenerate line segment in VP estimation: {segment}")
    return line / norm


def _line_intersection(line_a: np.ndarray, line_b: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Intersect two homogeneous lines and reject unstable or absurd points."""

    point = np.cross(line_a, line_b)
    if abs(float(point[2])) < config.VP_PARALLEL_INTERSECTION_EPS:
        return None
    point = point / point[2]
    max_abs = config.VP_MAX_COORD_MULTIPLIER * float(max(width, height))
    if not np.all(np.isfinite(point[:2])) or np.max(np.abs(point[:2])) > max_abs:
        return None
    return point[:2]


def _angular_residual_deg(segment: LineSegment, vp: np.ndarray) -> float:
    """Angular error between the segment direction and ray to the VP."""

    direction = np.array([segment.x2 - segment.x1, segment.y2 - segment.y1], dtype=np.float64)
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 0:
        return 180.0
    direction /= direction_norm

    midpoint = np.array([segment.midpoint_x, segment.midpoint_y], dtype=np.float64)
    ray = vp - midpoint
    ray_norm = np.linalg.norm(ray)
    if ray_norm <= 0:
        return 0.0
    ray /= ray_norm

    cosine = abs(float(np.clip(np.dot(direction, ray), -1.0, 1.0)))
    return math.degrees(math.acos(cosine))


def _candidate_pairs(num_lines: int, frame_index: int) -> list[tuple[int, int]]:
    """Return deterministic RANSAC line-pair samples."""

    all_pairs = list(itertools.combinations(range(num_lines), 2))
    if len(all_pairs) <= config.VP_EXHAUSTIVE_PAIR_LIMIT:
        return all_pairs

    rng = np.random.default_rng(config.RANDOM_SEED + frame_index)
    sampled: set[tuple[int, int]] = set()
    while len(sampled) < config.VP_RANSAC_ITERATIONS:
        a, b = rng.choice(num_lines, size=2, replace=False)
        pair = tuple(sorted((int(a), int(b))))
        sampled.add(pair)
    return sorted(sampled)


def _estimate_pairwise_vanishing_point(
    line_result: LineDetectionResult,
    width: int,
    height: int,
) -> VanishingPointEstimate:
    """Fallback dominant VP estimate using deterministic RANSAC over line pairs."""

    segments = line_result.accepted_segments
    for segment in segments:
        segment.vp_inlier = False
        segment.vp_residual_deg = None

    if len(segments) < 2:
        return VanishingPointEstimate(
            x=None,
            y=None,
            inside_image=None,
            inlier_count=0,
            inlier_ratio=0.0,
            residual_mean=None,
            residual_median=None,
            residual_max=None,
            geometrically_consistent=False,
        )

    homogeneous_lines = [_homogeneous_line(segment) for segment in segments]
    best_vp: np.ndarray | None = None
    best_inliers: list[int] = []
    best_residuals: list[float] = []
    best_score: tuple[int, float, float] | None = None

    for pair_a, pair_b in _candidate_pairs(len(segments), line_result.frame_index):
        vp = _line_intersection(homogeneous_lines[pair_a], homogeneous_lines[pair_b], width, height)
        if vp is None:
            continue
        residuals = [_angular_residual_deg(segment, vp) for segment in segments]
        inliers = [
            index
            for index, residual in enumerate(residuals)
            if residual <= config.VP_RANSAC_ANGULAR_THRESHOLD_DEG
        ]
        if not inliers:
            continue
        median = float(np.median([residuals[index] for index in inliers]))
        mean = float(np.mean([residuals[index] for index in inliers]))
        score = (len(inliers), -median, -mean)
        if best_score is None or score > best_score:
            best_score = score
            best_vp = vp
            best_inliers = inliers
            best_residuals = residuals

    if best_vp is None:
        return VanishingPointEstimate(
            x=None,
            y=None,
            inside_image=None,
            inlier_count=0,
            inlier_ratio=0.0,
            residual_mean=None,
            residual_median=None,
            residual_max=None,
            geometrically_consistent=False,
        )

    for index, segment in enumerate(segments):
        residual = float(best_residuals[index])
        segment.vp_residual_deg = residual
        segment.vp_inlier = index in best_inliers

    inlier_residuals = [best_residuals[index] for index in best_inliers]
    inlier_count = len(best_inliers)
    inlier_ratio = inlier_count / float(len(segments))
    inside = bool(0.0 <= best_vp[0] <= width - 1 and 0.0 <= best_vp[1] <= height - 1)
    consistent = (
        inlier_count >= config.VP_MIN_INLIER_LINES
        and inlier_ratio >= config.VP_MIN_INLIER_RATIO
    )
    return VanishingPointEstimate(
        x=round(float(best_vp[0]), 6),
        y=round(float(best_vp[1]), 6),
        inside_image=inside,
        inlier_count=inlier_count,
        inlier_ratio=round(float(inlier_ratio), 6),
        residual_mean=round(float(np.mean(inlier_residuals)), 6),
        residual_median=round(float(np.median(inlier_residuals)), 6),
        residual_max=round(float(np.max(inlier_residuals)), 6),
        geometrically_consistent=consistent,
    )


def _line_arrays(segments: list[LineSegment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Nx4 endpoints, homogeneous lines, and segment orientations."""

    lines = np.array(
        [[segment.x1, segment.y1, segment.x2, segment.y2] for segment in segments],
        dtype=np.float64,
    )
    p1 = np.column_stack((lines[:, :2], np.ones(len(segments), dtype=np.float64)))
    p2 = np.column_stack((lines[:, 2:], np.ones(len(segments), dtype=np.float64)))
    homogeneous = np.cross(p1, p2)
    orientations = np.arctan2(p1[:, 1] - p2[:, 1], p1[:, 0] - p2[:, 0])
    orientations[orientations < 0.0] += np.pi
    return lines, homogeneous, orientations


def _lu_vp_hypotheses(
    homogeneous_lines: np.ndarray,
    principal_point: np.ndarray,
    focal_length: float,
    frame_index: int,
) -> np.ndarray:
    """Generate orthogonal VP triplet hypotheses following Lu-style sampling."""

    num_lines = homogeneous_lines.shape[0]
    if num_lines < 2:
        return np.empty((0, 3, 3), dtype=np.float64)

    longitude_bins = int(round(360.0 / config.VP_LU_SPHERE_BIN_DEG))
    longitude_step = math.radians(config.VP_LU_SPHERE_BIN_DEG)
    latitudes = np.arange(longitude_bins, dtype=np.float64) * longitude_step
    hypotheses = np.zeros(
        (config.VP_LU_RANSAC_ITERATIONS * longitude_bins, 3, 3),
        dtype=np.float64,
    )
    rng = np.random.default_rng(config.RANDOM_SEED + frame_index)
    hypothesis_index = 0

    attempts = 0
    max_attempts = config.VP_LU_RANSAC_ITERATIONS * 20
    while hypothesis_index < config.VP_LU_RANSAC_ITERATIONS and attempts < max_attempts:
        attempts += 1
        idx1, idx2 = rng.choice(num_lines, size=2, replace=False)
        vp1_img = np.cross(homogeneous_lines[int(idx1)], homogeneous_lines[int(idx2)])
        if abs(float(vp1_img[2])) < config.VP_PARALLEL_INTERSECTION_EPS:
            continue

        vp1 = np.zeros(3, dtype=np.float64)
        vp1[:2] = vp1_img[:2] / vp1_img[2] - principal_point
        vp1[2] = focal_length
        norm = np.linalg.norm(vp1)
        if norm <= 0.0 or not np.isfinite(norm):
            continue
        vp1 /= norm

        kk = vp1[0] * np.sin(latitudes) + vp1[1] * np.cos(latitudes)
        phi = np.arctan2(-vp1[2], kk)
        vp2 = np.column_stack(
            [
                np.sin(phi) * np.sin(latitudes),
                np.sin(phi) * np.cos(latitudes),
                np.cos(phi),
            ]
        )
        vp2[np.abs(vp2[:, 2]) < config.VP_PARALLEL_INTERSECTION_EPS, 2] = config.VP_PARALLEL_INTERSECTION_EPS
        vp2 /= np.linalg.norm(vp2, axis=1, keepdims=True)
        vp2[vp2[:, 2] < 0.0, :] *= -1.0

        vp3 = np.cross(vp1, vp2)
        vp3[np.abs(vp3[:, 2]) < config.VP_PARALLEL_INTERSECTION_EPS, 2] = config.VP_PARALLEL_INTERSECTION_EPS
        vp3 /= np.linalg.norm(vp3, axis=1, keepdims=True)
        vp3[vp3[:, 2] < 0.0, :] *= -1.0

        start = hypothesis_index * longitude_bins
        stop = start + longitude_bins
        hypotheses[start:stop, 0, :] = vp1
        hypotheses[start:stop, 1, :] = vp2
        hypotheses[start:stop, 2, :] = vp3
        hypothesis_index += 1

    return hypotheses[: hypothesis_index * longitude_bins]


def _lu_sphere_grid(
    homogeneous_lines: np.ndarray,
    lengths: np.ndarray,
    orientations: np.ndarray,
    principal_point: np.ndarray,
    focal_length: float,
) -> np.ndarray:
    """Build the pair-intersection voting grid used to rank VP triplets."""

    bin_size = math.radians(config.VP_LU_SPHERE_BIN_DEG)
    num_bins_lat = int(round((math.pi / 2.0) / bin_size))
    num_bins_lon = int(round((2.0 * math.pi) / bin_size))
    combos = np.array(list(itertools.combinations(range(homogeneous_lines.shape[0]), 2)), dtype=np.int64)
    if combos.size == 0:
        return np.zeros((num_bins_lat, num_bins_lon), dtype=np.float64)

    intersections = np.cross(homogeneous_lines[combos[:, 0]], homogeneous_lines[combos[:, 1]])
    finite_mask = np.abs(intersections[:, 2]) >= config.VP_PARALLEL_INTERSECTION_EPS

    angle_delta = np.abs(orientations[combos[:, 0]] - orientations[combos[:, 1]])
    angle_delta = np.minimum(np.pi - angle_delta, angle_delta)
    angle_mask = angle_delta <= math.radians(config.VP_LU_PAIR_ORIENTATION_MAX_DEG)
    mask = finite_mask & angle_mask
    if not np.any(mask):
        return np.zeros((num_bins_lat, num_bins_lon), dtype=np.float64)

    intersections = intersections[mask]
    angle_delta = angle_delta[mask]
    combos = combos[mask]

    x = intersections[:, 0] / intersections[:, 2] - principal_point[0]
    y = intersections[:, 1] / intersections[:, 2] - principal_point[1]
    z = np.full_like(x, focal_length)
    norm = np.sqrt(x * x + y * y + z * z)
    lat = np.arccos(np.clip(z / norm, -1.0, 1.0))
    lon = np.arctan2(x, y) + math.pi

    lat_bin = np.clip((lat / bin_size).astype(np.int64), 0, num_bins_lat - 1)
    lon_bin = np.clip((lon / bin_size).astype(np.int64), 0, num_bins_lon - 1)
    bin_number = lat_bin * num_bins_lon + lon_bin
    weights = np.sqrt(lengths[combos[:, 0]] * lengths[combos[:, 1]]) * (np.sin(2.0 * angle_delta) + 0.2)
    grid = np.bincount(
        bin_number,
        weights=weights,
        minlength=num_bins_lat * num_bins_lon,
    ).reshape((num_bins_lat, num_bins_lon))

    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    return grid + cv2_filter2d(grid, kernel)


def cv2_filter2d(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small wrapper to keep the OpenCV import local to VP voting."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependencies are validated elsewhere
        raise ImportError("OpenCV is required for Lu-style VP voting.") from exc
    return cv2.filter2D(grid, -1, kernel)


def _best_lu_hypothesis(
    sphere_grid: np.ndarray,
    hypotheses: np.ndarray,
    principal_point: np.ndarray,
    focal_length: float,
) -> np.ndarray | None:
    """Select the VP triplet with the largest spherical-grid vote."""

    if hypotheses.size == 0 or sphere_grid.size == 0:
        return None

    bin_size = math.radians(config.VP_LU_SPHERE_BIN_DEG)
    num_bins_lat, num_bins_lon = sphere_grid.shape
    z = hypotheses[:, :, 2]
    mask = (np.abs(z) >= config.VP_PARALLEL_INTERSECTION_EPS) & (np.abs(z) <= 1.0)
    if not np.any(mask):
        return None

    ids = np.repeat(np.arange(hypotheses.shape[0])[:, None], 3, axis=1)[mask]
    lat = np.arccos(np.clip(z[mask], -1.0, 1.0))
    lon = np.arctan2(hypotheses[:, :, 0][mask], hypotheses[:, :, 1][mask]) + math.pi
    lat_bin = np.clip((lat / bin_size).astype(np.int64), 0, num_bins_lat - 1)
    lon_bin = np.clip((lon / bin_size).astype(np.int64), 0, num_bins_lon - 1)
    weights = sphere_grid[lat_bin, lon_bin]
    votes = np.bincount(ids, weights=weights, minlength=hypotheses.shape[0])
    if votes.size == 0 or float(np.max(votes)) <= 0.0:
        return None

    final_vps = hypotheses[int(np.argmax(votes))]
    vps_2d = focal_length * (final_vps[:, :2] / final_vps[:, 2][:, None])
    return vps_2d + principal_point


def _cluster_residuals_to_vps(
    segments: list[LineSegment],
    vps_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each segment to the closest VP and return angular residuals."""

    segment_lines = np.array(
        [[segment.x1, segment.y1, segment.x2, segment.y2] for segment in segments],
        dtype=np.float64,
    )
    midpoints = np.column_stack(
        [
            0.5 * (segment_lines[:, 0] + segment_lines[:, 2]),
            0.5 * (segment_lines[:, 1] + segment_lines[:, 3]),
        ]
    )
    directions = segment_lines[:, :2] - segment_lines[:, 2:]
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.divide(
        directions,
        direction_norms,
        out=np.zeros_like(directions),
        where=direction_norms > 0.0,
    )

    rays = vps_2d[:, None, :] - midpoints[None, :, :]
    ray_norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = np.divide(rays, ray_norms, out=np.zeros_like(rays), where=ray_norms > 0.0)
    dot = np.sum(directions[None, :, :] * rays, axis=2)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.arccos(dot)
    angles = np.minimum(np.pi - angles, angles)
    assignments = np.argmin(angles, axis=0)
    residuals = np.rad2deg(np.min(angles, axis=0))
    return assignments, residuals


def _center_occluder_outside_fraction(segment: LineSegment, width: int, height: int) -> float:
    """Return the fraction of a segment outside the central person/map region."""

    xs = np.linspace(segment.x1, segment.x2, config.LINE_CONFIDENCE_SAMPLE_COUNT)
    ys = np.linspace(segment.y1, segment.y2, config.LINE_CONFIDENCE_SAMPLE_COUNT)
    x_min = config.LINE_CENTER_OCCLUDER_X_MIN_FRACTION * width
    x_max = config.LINE_CENTER_OCCLUDER_X_MAX_FRACTION * width
    y_min = config.LINE_CENTER_OCCLUDER_Y_MIN_FRACTION * height
    y_max = config.LINE_CENTER_OCCLUDER_Y_MAX_FRACTION * height
    inside = (x_min <= xs) & (xs <= x_max) & (y_min <= ys) & (ys <= y_max)
    return float(np.mean(~inside))


def _estimate_lu_spherical_vanishing_point(
    line_result: LineDetectionResult,
    width: int,
    height: int,
) -> VanishingPointEstimate:
    """Estimate the dominant finite VP using Lu-style spherical voting."""

    segments = line_result.accepted_segments
    for segment in segments:
        segment.vp_inlier = False
        segment.vp_residual_deg = None

    if len(segments) < 2:
        return VanishingPointEstimate(
            x=None,
            y=None,
            inside_image=None,
            inlier_count=0,
            inlier_ratio=0.0,
            residual_mean=None,
            residual_median=None,
            residual_max=None,
            geometrically_consistent=False,
        )

    lines, homogeneous_lines, orientations = _line_arrays(segments)
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    principal_point = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    focal_length = config.VP_LU_FOCAL_LENGTH_MULTIPLIER * float(max(width, height))
    hypotheses = _lu_vp_hypotheses(
        homogeneous_lines,
        principal_point,
        focal_length,
        line_result.frame_index,
    )
    sphere_grid = _lu_sphere_grid(
        homogeneous_lines,
        lengths,
        orientations,
        principal_point,
        focal_length,
    )
    vps_2d = _best_lu_hypothesis(sphere_grid, hypotheses, principal_point, focal_length)
    if vps_2d is None or not np.all(np.isfinite(vps_2d)):
        return VanishingPointEstimate(
            x=None,
            y=None,
            inside_image=None,
            inlier_count=0,
            inlier_ratio=0.0,
            residual_mean=None,
            residual_median=None,
            residual_max=None,
            geometrically_consistent=False,
        )

    assignments, residuals = _cluster_residuals_to_vps(segments, vps_2d)
    support_weights = np.array(
        [
            lengths[index]
            * (0.25 + 0.75 * _center_occluder_outside_fraction(segment, width, height))
            for index, segment in enumerate(segments)
        ],
        dtype=np.float64,
    )
    max_abs = config.VP_MAX_COORD_MULTIPLIER * float(max(width, height))
    best_vp_index: int | None = None
    best_score: tuple[float, int, float] | None = None
    for vp_index, vp in enumerate(vps_2d):
        if np.max(np.abs(vp)) > max_abs:
            continue
        if abs(float(vp[1]) - height / 2.0) > config.VP_LU_VERTICAL_REJECTION_Y_MULTIPLIER * height:
            continue
        inliers = np.where(
            (assignments == vp_index)
            & (residuals <= config.VP_LU_CLUSTER_ANGULAR_THRESHOLD_DEG)
        )[0]
        support = float(np.sum(support_weights[inliers]))
        score = (support, int(inliers.size), -abs(float(vp[1]) - height / 2.0))
        if best_score is None or score > best_score:
            best_score = score
            best_vp_index = vp_index

    if best_vp_index is None:
        return VanishingPointEstimate(
            x=None,
            y=None,
            inside_image=None,
            inlier_count=0,
            inlier_ratio=0.0,
            residual_mean=None,
            residual_median=None,
            residual_max=None,
            geometrically_consistent=False,
        )

    inlier_indices = [
        index
        for index, assignment in enumerate(assignments)
        if assignment == best_vp_index
        and residuals[index] <= config.VP_LU_CLUSTER_ANGULAR_THRESHOLD_DEG
    ]
    for index, segment in enumerate(segments):
        residual = float(residuals[index])
        segment.vp_residual_deg = residual
        segment.vp_inlier = index in inlier_indices

    if not inlier_indices:
        return VanishingPointEstimate(
            x=None,
            y=None,
            inside_image=None,
            inlier_count=0,
            inlier_ratio=0.0,
            residual_mean=None,
            residual_median=None,
            residual_max=None,
            geometrically_consistent=False,
        )

    vp = vps_2d[best_vp_index]
    inlier_residuals = [float(residuals[index]) for index in inlier_indices]
    inlier_count = len(inlier_indices)
    inlier_ratio = inlier_count / float(len(segments))
    inside = bool(0.0 <= vp[0] <= width - 1 and 0.0 <= vp[1] <= height - 1)
    consistent = (
        inlier_count >= config.VP_MIN_INLIER_LINES
        and inlier_ratio >= config.VP_MIN_INLIER_RATIO
    )
    return VanishingPointEstimate(
        x=round(float(vp[0]), 6),
        y=round(float(vp[1]), 6),
        inside_image=inside,
        inlier_count=inlier_count,
        inlier_ratio=round(float(inlier_ratio), 6),
        residual_mean=round(float(np.mean(inlier_residuals)), 6),
        residual_median=round(float(np.median(inlier_residuals)), 6),
        residual_max=round(float(np.max(inlier_residuals)), 6),
        geometrically_consistent=consistent,
    )


def estimate_frame_vanishing_point(
    line_result: LineDetectionResult,
    width: int,
    height: int,
) -> VanishingPointEstimate:
    """Estimate the dominant receding/depth-direction VP for one frame."""

    lu_estimate = _estimate_lu_spherical_vanishing_point(line_result, width, height)
    if lu_estimate.x is not None:
        return lu_estimate
    return _estimate_pairwise_vanishing_point(line_result, width, height)


def estimate_video_global_vanishing_point(
    line_results: list[LineDetectionResult],
    width: int,
    height: int,
) -> VanishingPointEstimate:
    """Estimate one dominant video-level VP from all sampled-frame line evidence."""

    combined_segments: list[LineSegment] = []
    segment_id = 0
    for line_result in line_results:
        for segment in line_result.accepted_segments:
            combined_segments.append(
                LineSegment(
                    segment_id=segment_id,
                    x1=segment.x1,
                    y1=segment.y1,
                    x2=segment.x2,
                    y2=segment.y2,
                    length=segment.length,
                    angle_deg=segment.angle_deg,
                    midpoint_x=segment.midpoint_x,
                    midpoint_y=segment.midpoint_y,
                    edge_confidence=segment.edge_confidence,
                    edge_support_fraction=segment.edge_support_fraction,
                    accepted=segment.accepted,
                    rejection_reason=segment.rejection_reason,
                    family_label=segment.family_label,
                    object_or_scene_edge=segment.object_or_scene_edge,
                    selection_rationale=segment.selection_rationale,
                    selection_confidence=segment.selection_confidence,
                )
            )
            segment_id += 1

    combined = LineDetectionResult(
        frame_index=0,
        accepted_segments=combined_segments,
        rejected_segments=[],
    )
    return estimate_frame_vanishing_point(combined, width, height)


def estimate_frame_vanishing_point_against_reference(
    line_result: LineDetectionResult,
    width: int,
    height: int,
    reference_x: float,
    reference_y: float,
) -> VanishingPointEstimate:
    """Score one frame against a fixed video-level VP reference."""

    segments = line_result.accepted_segments
    vp = np.array([reference_x, reference_y], dtype=np.float64)
    for segment in segments:
        residual = _angular_residual_deg(segment, vp)
        segment.vp_residual_deg = float(residual)
        segment.vp_inlier = residual <= config.VP_REFERENCE_ANGULAR_THRESHOLD_DEG

    if not segments:
        return VanishingPointEstimate(
            x=round(float(reference_x), 6),
            y=round(float(reference_y), 6),
            inside_image=bool(0.0 <= reference_x <= width - 1 and 0.0 <= reference_y <= height - 1),
            inlier_count=0,
            inlier_ratio=0.0,
            residual_mean=None,
            residual_median=None,
            residual_max=None,
            geometrically_consistent=False,
        )

    inlier_residuals = [
        float(segment.vp_residual_deg)
        for segment in segments
        if segment.vp_inlier and segment.vp_residual_deg is not None
    ]
    inlier_count = len(inlier_residuals)
    inlier_ratio = inlier_count / float(len(segments))
    consistent = (
        inlier_count >= config.VP_MIN_INLIER_LINES
        and inlier_ratio >= config.VP_REFERENCE_MIN_INLIER_RATIO
    )
    inside = bool(0.0 <= reference_x <= width - 1 and 0.0 <= reference_y <= height - 1)
    return VanishingPointEstimate(
        x=round(float(reference_x), 6),
        y=round(float(reference_y), 6),
        inside_image=inside,
        inlier_count=inlier_count,
        inlier_ratio=round(float(inlier_ratio), 6),
        residual_mean=None if not inlier_residuals else round(float(np.mean(inlier_residuals)), 6),
        residual_median=None if not inlier_residuals else round(float(np.median(inlier_residuals)), 6),
        residual_max=None if not inlier_residuals else round(float(np.max(inlier_residuals)), 6),
        geometrically_consistent=consistent,
    )


def make_frame_diagnostics(
    sampled_frame: SampledFrame,
    edge_result: EdgeMapResult,
    line_result: LineDetectionResult,
    estimate: VanishingPointEstimate,
    annotated_frame_path: Path,
    project_root: Path,
) -> FrameVanishingPointDiagnostics:
    """Build the per-frame JSON diagnostics required by the benchmark."""

    coordinates = None if estimate.x is None or estimate.y is None else [estimate.x, estimate.y]
    return FrameVanishingPointDiagnostics(
        frame_index=sampled_frame.frame_index,
        timestamp_sec=round(sampled_frame.timestamp_sec, 6),
        number_detected_line_segments=len(line_result.all_segments),
        number_candidate_structural_line_segments=len(line_result.accepted_segments),
        number_vanishing_point_inliers=estimate.inlier_count,
        inlier_ratio=estimate.inlier_ratio,
        residual_mean=estimate.residual_mean,
        residual_median=estimate.residual_median,
        residual_max=estimate.residual_max,
        estimated_vanishing_point_coordinates=coordinates,
        vanishing_point_inside_image=estimate.inside_image,
        geometrically_consistent_with_one_dominant_vanishing_point=estimate.geometrically_consistent,
        raw_sampled_frame_path=display_path(sampled_frame.image_path, project_root),
        edge_map_path=display_path(edge_result.edge_png_path, project_root),
        annotated_frame_path=display_path(annotated_frame_path, project_root),
        accepted_line_segments=[segment.to_dict() for segment in line_result.accepted_segments],
        rejected_line_segments=[segment.to_dict() for segment in line_result.rejected_segments],
    )
