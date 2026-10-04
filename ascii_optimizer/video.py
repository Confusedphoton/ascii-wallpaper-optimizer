"""Timestamp-aware, bounded-memory video decoding through FFmpeg."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class VideoInfo:
    """Presentation timestamps relative to the first visible video frame."""

    frame_times_ms: tuple[float, ...]
    duration_ms: float

    def sampled_times(self, stride: int) -> tuple[float, ...]:
        return self.frame_times_ms[::stride]


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise ValueError(f"Video input requires {name}; install FFmpeg and make it available on PATH")
    return executable


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def probe_video(source: Path) -> VideoInfo:
    """Read timing metadata, without keeping decoded source images in memory."""
    _executable("ffmpeg")
    command = [_executable("ffprobe"), "-v", "error", "-select_streams", "v:0",
               "-show_frames", "-show_streams", "-show_format", "-show_entries",
               "stream=avg_frame_rate,r_frame_rate,duration:format=duration:"
               "frame=best_effort_timestamp_time,pkt_duration_time,duration_time",
               "-of", "json", str(source)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Could not read video: {result.stderr.strip() or source}")
    try:
        metadata = json.loads(result.stdout)
        stream = metadata["streams"][0]
        frames = metadata.get("frames", [])
    except (ValueError, KeyError, IndexError, TypeError):
        raise ValueError("Input does not contain a readable video stream") from None
    if not frames:
        raise ValueError("Input video contains no decodable frames")
    try:
        rate = float(Fraction(stream.get("avg_frame_rate", "0/1")))
        if not math.isfinite(rate) or rate <= 0:
            rate = float(Fraction(stream.get("r_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError, TypeError):
        rate = 0
    step = 1000 / rate if rate > 0 else None
    timestamps = [_number(frame.get("best_effort_timestamp_time")) for frame in frames]
    if any(value is None for value in timestamps):
        if step is None:
            raise ValueError("Video has no usable frame timestamps or frame rate")
        times = tuple(index * step for index in range(len(frames)))
    else:
        start = timestamps[0]
        times = tuple((value - start) * 1000 for value in timestamps)
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("Video frame timestamps must increase in presentation order")
    final_duration = _number(frames[-1].get("duration_time"))
    if final_duration is None or final_duration <= 0:
        final_duration = _number(frames[-1].get("pkt_duration_time"))
    tail_ms = (final_duration * 1000 if final_duration is not None and final_duration > 0
               else times[-1] - times[-2] if len(times) > 1 else step)
    duration = _number(stream.get("duration"))
    if duration is None or duration * 1000 <= times[-1]:
        # Container duration may include a longer audio track. Prefer the
        # decoded video timeline whenever a final frame duration is available.
        duration = ((times[-1] + tail_ms) / 1000 if tail_ms is not None and tail_ms > 0
                    else _number(metadata.get("format", {}).get("duration")))
    duration_ms = duration * 1000 if duration is not None else 0
    if final_duration is not None and final_duration > 0:
        duration_ms = max(duration_ms, times[-1] + final_duration * 1000)
    if duration_ms <= times[-1]:
        if tail_ms is None or tail_ms <= 0:
            raise ValueError("Could not determine video duration")
        duration_ms = times[-1] + tail_ms
    return VideoInfo(times, duration_ms)


def frame_durations(info: VideoInfo, stride: int, transition_frames: int) -> list[float]:
    """Subdivide endpoint intervals while preserving the original clip duration."""
    sampled = info.sampled_times(stride)
    if not sampled:
        raise ValueError("Input video contains no sampled frames")
    intervals = []
    for left, right in zip(sampled, sampled[1:]):
        intervals.extend([(right - left) / (transition_frames + 1)] * (transition_frames + 1))
    intervals.append(info.duration_ms - sampled[-1])
    return intervals


def decode_frames(source: Path, size: tuple[int, int], crop: str,
                  grayscale: bool, stride: int) -> Iterator[np.ndarray]:
    """Yield sampled RGB targets; FFmpeg's pipe bounds decoded-frame memory."""
    width, height = size
    filters = [f"select=not(mod(n\\,{stride}))"]
    if crop == "fill":
        filters.extend([f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
                        f"crop={width}:{height}"])
    elif crop == "fit":
        filters.extend([f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos",
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"])
    else:
        raise ValueError(f"Unknown crop mode: {crop}")
    filters.append("setsar=1")
    command = [_executable("ffmpeg"), "-v", "error", "-nostdin", "-i", str(source),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", ",".join(filters),
               "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    frame_bytes = width * height * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                raw = bytearray()
                while len(raw) < frame_bytes:
                    chunk = process.stdout.read(frame_bytes - len(raw))
                    if not chunk:
                        break
                    raw.extend(chunk)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise ValueError("FFmpeg returned an incomplete video frame")
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                if grayscale:
                    rgb = np.asarray(ImageOps.grayscale(Image.fromarray(rgb)).convert("RGB"))
                yield rgb.astype(np.float32) / 255
            if process.wait() != 0:
                errors.seek(0)
                detail = errors.read().decode(errors="replace").strip()
                raise ValueError(f"Could not decode video: {detail or source}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
