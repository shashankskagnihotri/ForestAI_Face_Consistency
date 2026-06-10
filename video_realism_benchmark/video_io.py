"""Video validation, metadata probing, and timestamp extraction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
from typing import Sequence

try:
    import av
except ImportError as exc:  # pragma: no cover - exercised only on broken envs
    raise ImportError(
        "PyAV is required for deterministic video decoding. Install dependencies "
        "with `pip install -r requirements.txt` inside the conda environment."
    ) from exc


@dataclass(frozen=True)
class VideoMetadata:
    """Validated metadata and per-frame timestamps for one input video."""

    video_path: Path
    filename: str
    duration_seconds: float
    frame_count: int
    fps: float
    width: int
    height: int
    frame_timestamps_sec: tuple[float, ...]
    timestamp_source: str


def _fraction_to_float(value: Fraction | None) -> float | None:
    """Convert a PyAV rate/time-base Fraction into a positive float."""

    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        return None
    return result


def _stream_duration_seconds(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> float | None:
    """Read the most reliable available duration metadata."""

    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    if container.duration is not None:
        duration = float(container.duration / av.time_base)
        if math.isfinite(duration) and duration > 0:
            return duration
    return None


def _positive_median_delta(timestamps: Sequence[float], fps: float) -> float:
    """Estimate a final-frame duration from timestamp deltas or FPS."""

    deltas = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
        if timestamps[index] > timestamps[index - 1]
    ]
    if not deltas:
        return 1.0 / fps
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _normalize_timestamps(raw_timestamps: list[float]) -> tuple[float, ...]:
    """Normalize timestamps so the first decoded frame is time zero."""

    first = raw_timestamps[0]
    normalized = tuple(round(value - first, 9) for value in raw_timestamps)
    for index, timestamp in enumerate(normalized):
        if not math.isfinite(timestamp) or timestamp < -1e-9:
            raise RuntimeError(f"Invalid timestamp for frame {index}: {timestamp}")
        if index > 0 and timestamp < normalized[index - 1] - 1e-9:
            raise RuntimeError(
                "Decoded frame timestamps are not monotonic; the input video "
                "cannot be evaluated deterministically."
            )
    return normalized


def probe_video(video_path: Path) -> VideoMetadata:
    """Validate a video and decode it once to obtain exact frame timestamps."""

    if not video_path.exists():
        raise FileNotFoundError(f"Input video path does not exist: {video_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video path is not a file: {video_path}")

    try:
        container = av.open(str(video_path))
    except Exception as exc:
        raise RuntimeError(f"Could not open video with PyAV: {video_path}") from exc

    try:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        if not video_streams:
            raise RuntimeError(f"No video stream found in file: {video_path}")
        stream = video_streams[0]

        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")

        fps = _fraction_to_float(stream.average_rate) or _fraction_to_float(stream.base_rate)
        if fps is None:
            raise RuntimeError(
                f"Could not determine a positive FPS for {video_path}; timestamps "
                "cannot be validated."
            )

        metadata_duration = _stream_duration_seconds(container, stream)

        raw_timestamps: list[float] = []
        missing_pts_count = 0
        decoded_with_pts_count = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                raw_timestamps.append(float(frame.pts * frame.time_base))
                decoded_with_pts_count += 1
            else:
                raw_timestamps.append(frame_index / fps)
                missing_pts_count += 1

        if not raw_timestamps:
            raise RuntimeError(f"Video has no decodable frames: {video_path}")
        if missing_pts_count and decoded_with_pts_count:
            raise RuntimeError(
                "Some decoded frames have timestamps and some do not; refusing to "
                "mix timestamp sources."
            )

        timestamps = _normalize_timestamps(raw_timestamps)
        frame_duration = _positive_median_delta(timestamps, fps)
        decoded_duration = timestamps[-1] + frame_duration
        duration_seconds = metadata_duration if metadata_duration is not None else decoded_duration
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise RuntimeError(f"Invalid video duration for {video_path}: {duration_seconds}")

        timestamp_source = "packet_pts" if decoded_with_pts_count else "frame_index_divided_by_fps"
        return VideoMetadata(
            video_path=video_path.resolve(),
            filename=video_path.name,
            duration_seconds=round(float(duration_seconds), 6),
            frame_count=len(timestamps),
            fps=float(fps),
            width=width,
            height=height,
            frame_timestamps_sec=timestamps,
            timestamp_source=timestamp_source,
        )
    finally:
        container.close()
