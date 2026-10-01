from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import av


@dataclass(frozen=True)
class VideoTimeline:
    start: float
    duration: float


class VideoReader:
    def interval_frames(self, path, windows):
        """Decode only merged native-PTS windows; release the container on close."""
        merged = []
        for start, stop in sorted(windows):
            if stop <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("В файле нет видеопотока")
            container.streams.video[0].thread_count = 1
            for start, stop in merged:
                container.seek(int(start * av.time_base), backward=True)
                for frame in container.decode(video=0):
                    if frame.time is None:
                        continue
                    time = float(frame.time)
                    if time < start:
                        continue
                    if time > stop:
                        break
                    yield time, frame.to_ndarray(format="bgr24")

    def frame_at(self, path, timestamp):
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("В файле нет видеопотока")
            container.streams.video[0].thread_count = 1
            container.seek(int(timestamp * av.time_base), backward=True)
            for frame in container.decode(video=0):
                if frame.time is None:
                    continue
                if abs(float(frame.time) - timestamp) < 1e-6:
                    return frame.to_ndarray(format="bgr24")
                if float(frame.time) > timestamp + 1e-6:
                    break
        raise ValueError(f"Исходный кадр {timestamp:.6f} сек не найден")

    def timeline(self, path, scanning=None):
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("В файле нет видеопотока")
            stream = container.streams.video[0]
            start = float((stream.start_time or 0) * stream.time_base)
            duration = (
                float(stream.duration * stream.time_base) if stream.duration is not None else 0
            )
            if not isfinite(duration) or duration <= 0:
                duration = float(container.duration or 0) / av.time_base
            if isfinite(duration) and duration > 0:
                return VideoTimeline(start, duration)
            if scanning:
                scanning()
            stream.thread_count = 1
            first = last = None
            for frame in container.decode(video=0):
                if frame.time is None:
                    continue
                timestamp = float(frame.time)
                if not isfinite(timestamp) or (last is not None and timestamp < last):
                    raise ValueError("Некорректные временные отметки видео")
                if first is None:
                    first = timestamp
                last = timestamp
            if first is None or last <= first:
                raise ValueError("Невозможно определить длительность видео")
            return VideoTimeline(first, last - first)

    def frames(self, path: Path, sample_rate=2):
        """Decode sequentially; presentation timestamps also support variable FPS."""
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("В файле нет видеопотока")
            container.streams.video[0].thread_count = 1
            next_time = 0.0
            for frame in container.decode(video=0):
                if frame.time is None:
                    continue
                time = float(frame.time)
                if time + 1e-6 < next_time:
                    continue
                yield time, frame.to_ndarray(format="bgr24")
                next_time = time + 1 / sample_rate
