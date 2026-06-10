"""Deterministic frame-index selection and sampled-frame extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import cv2
import numpy as np

try:
    import av
except ImportError as exc:  # pragma: no cover - exercised only on broken envs
    raise ImportError(
        "PyAV is required for frame extraction. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from . import config


@dataclass(frozen=True)
class SampledFrame:
    """A raw sampled frame saved under the result directory."""

    frame_index: int
    timestamp_sec: float
    image_path: Path
    width: int
    height: int


def sample_frame_indices(frame_count: int) -> list[int]:
    """Sample every 16th frame plus three reproducible random frames."""

    if frame_count <= 0:
        raise ValueError("Cannot sample frames from a video with no frames.")

    stride_indices = list(range(0, frame_count, config.FRAME_SAMPLING_STRIDE))
    rng = random.Random(config.RANDOM_SEED)
    random_count = min(config.NUM_RANDOM_FRAMES, frame_count)
    random_indices = rng.sample(range(frame_count), k=random_count)
    return sorted(set(stride_indices + random_indices))


def _sampled_frame_filename(frame_index: int, timestamp_sec: float) -> str:
    """Create a deterministic sampled-frame filename."""

    return f"frame_{frame_index:06d}_t_{timestamp_sec:.3f}s.png"


def extract_sampled_frames(
    video_path: Path,
    sample_indices: Sequence[int],
    frame_timestamps_sec: Sequence[float],
    output_dir: Path,
) -> list[SampledFrame]:
    """Decode and save selected raw frames using PyAV."""

    if not sample_indices:
        raise RuntimeError("Frame sampling produced no frame indices.")
    invalid = [index for index in sample_indices if index < 0 or index >= len(frame_timestamps_sec)]
    if invalid:
        raise RuntimeError(f"Sampled frame indices are out of bounds: {invalid}")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_indices = set(sample_indices)
    saved: dict[int, SampledFrame] = {}

    try:
        container = av.open(str(video_path))
    except Exception as exc:
        raise RuntimeError(f"Could not reopen video for sampled-frame extraction: {video_path}") from exc

    try:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        if not video_streams:
            raise RuntimeError(f"No video stream found while extracting frames: {video_path}")
        stream = video_streams[0]

        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in target_indices:
                continue
            rgb_frame = frame.to_ndarray(format="rgb24")
            if rgb_frame.size == 0:
                raise RuntimeError(f"Decoded sampled frame is empty: frame {frame_index}")

            timestamp = float(frame_timestamps_sec[frame_index])
            image_path = output_dir / _sampled_frame_filename(frame_index, timestamp)
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(image_path), bgr_frame):
                raise RuntimeError(f"OpenCV failed to write sampled frame: {image_path}")

            height, width = rgb_frame.shape[:2]
            saved[frame_index] = SampledFrame(
                frame_index=frame_index,
                timestamp_sec=timestamp,
                image_path=image_path,
                width=int(width),
                height=int(height),
            )
    finally:
        container.close()

    missing = sorted(target_indices - set(saved))
    if missing:
        raise RuntimeError(f"Could not decode sampled frame indices: {missing}")
    if not saved:
        raise RuntimeError("No sampled frames were saved.")

    return [saved[index] for index in sorted(saved)]


def load_sampled_frame_rgb(sampled_frame: SampledFrame) -> np.ndarray:
    """Load a saved sampled frame as RGB for downstream deterministic CV."""

    bgr = cv2.imread(str(sampled_frame.image_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise RuntimeError(f"Failed to read sampled frame image: {sampled_frame.image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
