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
