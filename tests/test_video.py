from types import SimpleNamespace

import pytest

from domino_video.video import VideoReader


class Container:
    def __init__(self, duration=None, container_duration=None, times=(10, 11, 12)):
        self.streams = SimpleNamespace(
            video=[SimpleNamespace(duration=duration, time_base=0.5, start_time=20, thread_count=0)]
        )
        self.duration = container_duration
        self.times = times
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def decode(self, **kwargs):
        return (SimpleNamespace(time=t) for t in self.times)


@pytest.mark.parametrize(
    "duration,container_duration,expected", [(8, 9000000, 4), (None, 9000000, 9), (None, None, 2)]
)
def test_duration_uses_stream_then_container_then_pts(
    monkeypatch, duration, container_duration, expected
):
    container = Container(duration, container_duration)
    monkeypatch.setattr("domino_video.video.av.open", lambda path: container)
    phases = []
    timeline = VideoReader().timeline("video.mp4", lambda: phases.append("scan"))
    assert (timeline.start, timeline.duration) == (10, expected)
    assert phases == ([] if duration or container_duration else ["scan"])
    assert container.closed


@pytest.mark.parametrize("times", [(), (None,), (2,), (2, 2), (3, 2), (float("nan"), 4)])
def test_rejects_unusable_pts_range(monkeypatch, times):
    container = Container(times=times)
    monkeypatch.setattr("domino_video.video.av.open", lambda path: container)
    with pytest.raises(ValueError):
        VideoReader().timeline("video.mp4")
    assert container.closed


@pytest.mark.parametrize("timestamp,found", [(10.333333, True), (10.4, False), (20, False)])
def test_frame_at_requires_exact_pts_and_closes_decoder(monkeypatch, timestamp, found):
    class Frame:
        def __init__(self, time):
            self.time = time

        def to_ndarray(self, format):
            assert format == "bgr24"
            return "exact image"

    container = Container(times=(10, 10.333333, 10.666667))
    seek_calls = []
    container.seek = lambda offset, **kwargs: seek_calls.append((offset, kwargs))
    container.decode = lambda **kwargs: (Frame(t) for t in container.times)
    monkeypatch.setattr("domino_video.video.av.open", lambda path: container)
    if found:
        assert VideoReader().frame_at("video.mp4", timestamp) == "exact image"
    else:
        with pytest.raises(ValueError, match="кадр"):
            VideoReader().frame_at("video.mp4", timestamp)
    assert seek_calls == [(int(timestamp * 1000000), dict(backward=True))]
    assert container.closed


def test_decoder_allows_frame_parallelism(monkeypatch):
    container = Container(times=(0, 1, 2))
    monkeypatch.setattr("domino_video.video.av.open", lambda path: container)
    VideoReader().timeline("video.mp4")
    assert container.streams.video[0].thread_type == "AUTO"


@pytest.mark.parametrize("device,available", [("cpu", True), ("gpu", True), ("gpu", False)])
def test_decoder_selection_probes_once_and_closes(monkeypatch, device, available):
    opened, messages = [], []

    def open_video(path, **kwargs):
        container = Container(times=(0, 1, 2))
        container.streams.video[0].codec_context = SimpleNamespace(
            is_hwaccel=True, format=SimpleNamespace(name="yuv420p")
        )
        frame = SimpleNamespace(to_ndarray=lambda **k: None)
        frame.reformat = lambda **k: frame
        container.decode = lambda **k: iter([frame])
        container.duration = 1000000
        opened.append((container, kwargs))
        if kwargs and not available:
            raise RuntimeError("no decoder")
        return container

    monkeypatch.setattr("domino_video.video.av.open", open_video)
    reader = VideoReader(workers=4, device=device, message=messages.append)
    for _ in range(2):
        reader.timeline("video.mp4")
    accelerated = [kw for _, kw in opened if kw]
    assert len(accelerated) == (0 if device == "cpu" else 3 if available else 1)
    assert all(c.closed for c, kw in opened if not kw or available)
    assert len(messages) == 1
    assert ("GPU (CUDA)" in messages[0]) == (device != "cpu" and available)
    if not available:
        assert "no decoder" in messages[0]
    for container, kw in opened:
        if not kw:
            assert container.streams.video[0].thread_count == 4
            assert container.streams.video[0].thread_type == "AUTO"


@pytest.mark.parametrize("stop", ["normal", "close", "missing", "error"])
def test_selected_frames_reuse_decoder_and_preserve_exact_pts(monkeypatch, stop):
    opened, seeks = [], []
    container = Container(times=(0, 0.5, 1, 1.5, 8, 8.5))
    container.seek = lambda offset, **kwargs: seeks.append(offset)

    def decode(**kwargs):
        for t in container.times:
            if stop == "error" and t == 1:
                raise RuntimeError("decode failed")
            yield SimpleNamespace(time=t, to_ndarray=lambda format, t=t: f"image {t}")

    container.decode = decode
    monkeypatch.setattr("domino_video.video.av.open", lambda path: opened.append(path) or container)
    times = [8.5, 0.5, 1.5, 0.5] if stop != "missing" else [0.5, 1.25]
    frames = VideoReader().selected_frames("video.mp4", times)
    if stop == "close":
        assert next(frames) == (0.5, "image 0.5")
        frames.close()
    elif stop in ("missing", "error"):
        with pytest.raises(ValueError if stop == "missing" else RuntimeError):
            list(frames)
    else:
        assert list(frames) == [(t, f"image {t}") for t in (0.5, 1.5, 8.5)]
        assert seeks == [500000, 8500000]
    assert opened == ["video.mp4"]
    assert container.closed


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_gpu_decode_failure_after_probe_does_not_restart_on_cpu(monkeypatch, failure):
    opened = []

    def open_video(path, **kwargs):
        container = Container(container_duration=1000000)
        container.streams.video[0].codec_context = SimpleNamespace(
            format=SimpleNamespace(name="yuv420p"), is_hwaccel=True
        )
        frame = SimpleNamespace(time=0, to_ndarray=lambda **k: "image")
        frame.reformat = lambda **k: frame
        number = len(opened)

        def decode(**kwargs):
            yield frame
            if number == 2:
                raise failure("decoder stopped")

        container.decode = decode
        opened.append((container, kwargs))
        return container

    monkeypatch.setattr("domino_video.video.av.open", open_video)
    frames = VideoReader(device="gpu").frames("video.mp4")
    assert next(frames) == (0, "image")
    with pytest.raises(failure, match="decoder stopped"):
        next(frames)
    assert len(opened) == 3
    assert opened[1][1] and opened[2][1]
    assert all(c.closed for c, _ in opened)


def test_gpu_probe_interrupt_closes_without_cpu_fallback(monkeypatch):
    opened = []

    def open_video(path, **kwargs):
        container = Container(container_duration=1000000)
        container.streams.video[0].codec_context = SimpleNamespace(
            format=SimpleNamespace(name="yuv420p")
        )

        def decode(**kwargs):
            raise KeyboardInterrupt

        container.decode = decode
        opened.append(container)
        return container

    monkeypatch.setattr("domino_video.video.av.open", open_video)
    with pytest.raises(KeyboardInterrupt):
        list(VideoReader(device="auto").frames("video.mp4"))
    assert len(opened) == 2 and all(c.closed for c in opened)


def test_real_threaded_decoder_matches_single_thread_pts_and_pixels(tmp_path):
    import av
    import numpy as np

    path = tmp_path / "frames.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        for i in range(40):
            image = np.full((64, 96, 3), i * 5, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    baseline = dict(VideoReader(1).frames(path, sample_rate=10))
    parallel = dict(VideoReader(4).frames(path, sample_rate=10))
    assert len(baseline) == 40 and baseline.keys() == parallel.keys()
    assert all(np.array_equal(image, parallel[t]) for t, image in baseline.items())
    reader = VideoReader(4)
    times = [0, 0.1, 1.5, 1.6, 3.9]
    selected = dict(reader.selected_frames(path, times))
    assert list(selected) == times
    assert all(np.array_equal(baseline[t], selected[t]) for t in times)
    dense = dict(reader.interval_frames(path, [(0, 0.2), (1.5, 1.7)]))
    assert list(dense) == [0, 0.1, 0.2, 1.5, 1.6, 1.7]
    assert all(np.array_equal(baseline[t], image) for t, image in dense.items())


@pytest.mark.parametrize("gpu_seconds,use_gpu", [(0.05, True), (0.2, False), (0.3, False)])
def test_auto_decoder_uses_gpu_only_if_faster(monkeypatch, gpu_seconds, use_gpu):
    messages, opened, probes = [], [], []

    def open_video(path, **kwargs):
        container = Container(container_duration=1000000)
        container.streams.video[0].codec_context = SimpleNamespace(
            format=SimpleNamespace(name="yuv420p"), is_hwaccel=True
        )
        frame = SimpleNamespace(to_ndarray=lambda **k: None)
        frame.reformat = lambda **k: frame
        container.decode = lambda **k: iter([frame])
        opened.append((container, kwargs))
        return container

    def probe(self, path, hardware):
        probes.append(hardware)
        return gpu_seconds if hardware else 0.2

    monkeypatch.setattr("domino_video.video.av.open", open_video)
    monkeypatch.setattr(VideoReader, "_probe_speed", probe, raising=False)
    reader = VideoReader(4, "auto", messages.append)
    reader.timeline("video.mp4")
    assert probes == [False, True]
    assert bool(opened[-1][1]) is use_gpu
    assert ("GPU (CUDA)" in messages[0]) is use_gpu
    if not use_gpu:
        assert "быстрее" in messages[0] and "0.200" in messages[0]
    assert all(c.closed for c, _ in opened)


def test_auto_probes_same_bounded_workload_and_normalizes_gpu_pixels(monkeypatch):
    containers, decoded, downloaded, reformatted = [], [], [], []
    ticks = iter([0, 1, 2, 2.5])
    monkeypatch.setattr("domino_video.video.perf_counter", lambda: next(ticks))

    def open_video(path, **kwargs):
        number = len(containers)
        container = Container(container_duration=1000000)
        container.streams.video[0].codec_context = SimpleNamespace(
            format=SimpleNamespace(name="yuv420p"), is_hwaccel=True
        )
        containers.append(container)
        decoded.append([])
        downloaded.append([])

        def decode(**kwargs):
            for i in range(500):
                frame = SimpleNamespace(time=i / 10)
                frame.to_ndarray = lambda format, i=i: downloaded[number].append(i)
                frame.reformat = lambda format, f=frame: reformatted.append(format) or f
                decoded[number].append(i)
                yield frame

        container.decode = decode
        return container

    monkeypatch.setattr("domino_video.video.av.open", open_video)
    VideoReader(4, "auto").timeline("video.mp4")
    assert [len(rows) for rows in decoded] == [0, 1, 120, 120, 0]
    assert downloaded[2] == downloaded[3] == list(range(0, 120, 5))
    assert reformatted == ["yuv420p"] * 25
    assert all(c.closed for c in containers)
