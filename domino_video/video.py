from contextlib import closing, contextmanager
from dataclasses import dataclass
from itertools import islice
from math import isfinite
from pathlib import Path
from time import perf_counter

import av
from av.video.reformatter import VideoReformatter


@dataclass(frozen=True)
class VideoTimeline:
    start: float
    duration: float


class VideoReader:
    def __init__(self, workers=1, device="cpu", message=None):
        self._workers = workers
        self._device = device
        self._message = message
        self._source = None
        self._hardware = False
        self._pixel_format = None
        self._reformatter = None

    def _select_decoder(self, path):
        if self._source == str(path):
            return
        self._source = None
        self._hardware = False
        reason = ""
        if self._device != "cpu":
            try:
                with self._container(path) as software:
                    self._pixel_format = software.streams.video[0].codec_context.format.name
                with self._container(path, hardware=True) as container:
                    frame = next(container.decode(video=0))
                    frame.reformat(format=self._pixel_format).to_ndarray(format="bgr24")
                    if not container.streams.video[0].codec_context.is_hwaccel:
                        raise RuntimeError("аппаратный декодер не активирован")
                self._hardware = True
                if self._device == "auto":
                    cpu_time = self._probe_speed(path, hardware=False)
                    gpu_time = self._probe_speed(path, hardware=True)
                    if gpu_time >= cpu_time:
                        self._hardware = False
                        reason = (
                            f"; GPU не быстрее CPU на пробе "
                            f"(CPU {cpu_time:.3f} с, GPU {gpu_time:.3f} с)"
                        )
            except Exception as error:
                self._hardware = False
                reason = f"; CUDA недоступна: {type(error).__name__}: {error}"
        self._source = str(path)
        if self._message:
            backend = "GPU (CUDA)" if self._hardware else f"CPU, потоков: {self._workers}{reason}"
            self._message(f"Декодирование: {backend}")

    @contextmanager
    def _container(self, path, hardware=False):
        options = {}
        if hardware:
            from av.codec.hwaccel import HWAccel

            options["hwaccel"] = HWAccel(
                "cuda", device="0", allow_software_fallback=False, is_hw_owned=True
            )
        with av.open(str(path), **options) as container:
            if not container.streams.video:
                raise ValueError("В файле нет видеопотока")
            stream = container.streams.video[0]
            stream.thread_count = 1 if hardware else self._workers
            stream.thread_type = "AUTO"
            yield container

    @contextmanager
    def _open(self, path):
        self._select_decoder(path)
        with self._container(path, self._hardware) as container:
            yield container

    def _probe_speed(self, path, hardware):
        started = perf_counter()
        with self._container(path, hardware) as container:
            next_time = 0.0
            # Bounded sample, same 2 fps workload as the main stone pass.
            for frame in islice(container.decode(video=0), 120):
                if frame.time is not None and float(frame.time) + 1e-6 >= next_time:
                    self._image(frame, hardware)
                    next_time = float(frame.time) + 0.5
        return perf_counter() - started

    def _image(self, frame, hardware=None):
        accelerated = self._hardware if hardware is None else hardware
        if accelerated:
            # CUDA returns NV12 surfaces; converting them straight to BGR uses
            # different chroma interpolation than the source's planar format.
            # Download only sampled frames and preserve the CPU pixel path.
            frame = frame.reformat(format=self._pixel_format)
        # Reuse the swscale context instead of allocating one for every native
        # frame. Keep the same default conversion and source pixel format.
        if self._reformatter is None:
            self._reformatter = VideoReformatter()
        return self._reformatter.reformat(frame, format="bgr24").to_ndarray(format="bgr24")

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
        with self._open(path) as container:
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
                    yield time, self._image(frame)

    def frame_at(self, path, timestamp):
        with closing(self.selected_frames(path, [timestamp])) as frames:
            return next(frames)[1]

    def selected_frames(self, path, timestamps):
        targets = sorted(set(timestamps))
        if not targets:
            return
        with self._open(path) as container:
            decoded = None
            previous = None
            for timestamp in targets:
                # Nearby OCR requests share decoded frames. One second is a
                # seek-cost heuristic, independent of exact-PTS validation.
                if previous is None or timestamp - previous > 1:
                    container.seek(int(timestamp * av.time_base), backward=True)
                    decoded = iter(container.decode(video=0))
                for frame in decoded:
                    if frame.time is None:
                        continue
                    time = float(frame.time)
                    if abs(time - timestamp) < 1e-6:
                        yield timestamp, self._image(frame)
                        break
                    if time > timestamp + 1e-6:
                        raise ValueError(f"Исходный кадр {timestamp:.6f} сек не найден")
                else:
                    raise ValueError(f"Исходный кадр {timestamp:.6f} сек не найден")
                previous = timestamp

    def timeline(self, path, scanning=None):
        with self._open(path) as container:
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
        with self._open(path) as container:
            next_time = 0.0
            for frame in container.decode(video=0):
                if frame.time is None:
                    continue
                time = float(frame.time)
                if time + 1e-6 < next_time:
                    continue
                yield time, self._image(frame)
                next_time = time + 1 / sample_rate
