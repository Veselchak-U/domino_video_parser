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
