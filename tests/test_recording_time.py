from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from domino_video.recording_time import RecordingTimeResolver

NOW = datetime(2026, 9, 28, 15, 4, 3, 7999, tzinfo=timezone.utc)


def local_time(value):
    return value.astimezone(timezone(timedelta(hours=3)))


@pytest.fixture
def metadata(monkeypatch):
    state = {"closed": False, "container": {}, "stream": {}}

    class Container:
        def __enter__(self):
            self.metadata = state["container"]
            self.streams = SimpleNamespace(video=[SimpleNamespace(metadata=state["stream"])])
            return self

        def __exit__(self, *args):
            state["closed"] = True

    monkeypatch.setattr("av.open", lambda path: Container())
    return state


@pytest.mark.parametrize(
    "name",
    [
        "Record_2026-09-28-13-58.mp4",
        "Record_2026-09-28-13-58-19_hash.mp4",
        "2026-09-28-13-58_2026-09-28-13-58.mp4",
    ],
)
def test_prefers_filename_without_opening_video(monkeypatch, name):
    def unexpected(*args):
        pytest.fail("Metadata must not be read")

    monkeypatch.setattr("av.open", unexpected)
    result = RecordingTimeResolver(to_local=local_time).resolve(Path(name))
    assert result.prefix == "2026-09-28-13-58"
    assert result.source == "filename"
    assert result.original == "2026-09-28-13-58"


@pytest.mark.parametrize(
    "name",
    [
        "2026-02-30-13-58.mp4",
        "2026-09-28-25-58.mp4",
        "12026-09-28-13-58.mp4",
        "2026-09-28-13-589.mp4",
        "2026-09-28-13-58/plain.mp4",
    ],
)
def test_ignores_invalid_dates_digit_boundaries_and_directory(name, metadata):
    metadata["container"]["creation_time"] = "2026-09-28T11:05:07Z"
    result = RecordingTimeResolver(to_local=local_time).resolve(Path(name))
    assert result.prefix == "2026-09-28-14-05"
    assert result.source == "container.creation_time"
    assert metadata["closed"]


def test_rejects_distinct_filename_dates(metadata):
    with pytest.raises(ValueError, match="Неоднознач"):
        RecordingTimeResolver(to_local=local_time).resolve(
            Path("2026-09-28-13-58_2026-09-29-13-58.mp4")
        )
    assert not metadata["closed"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-09-28T14:05:07+03:00", "2026-09-28-14-05"),
        ("2026-09-28T11:05:07.000000Z", "2026-09-28-14-05"),
        ("2026-09-28T14:05", "2026-09-28-14-05"),
    ],
)
def test_normalizes_metadata_timezone(value, expected, metadata):
    metadata["container"]["creation_time"] = value
    metadata["stream"]["creation_time"] = "2000-01-01T00:00:00Z"
    result = RecordingTimeResolver(to_local=local_time).resolve(Path("video.mp4"))
    assert result.prefix == expected
    assert result.original == value
    assert metadata["closed"]


@pytest.mark.parametrize("container", [None, "invalid", "2026-09-28"])
def test_uses_stream_when_container_has_no_valid_datetime(container, metadata):
    if container is not None:
        metadata["container"]["creation_time"] = container
    metadata["stream"]["creation_time"] = "2026-09-28T11:05:07Z"
    result = RecordingTimeResolver(to_local=local_time).resolve(Path("video.mp4"))
    assert result.source == "video.creation_time"
    assert result.prefix == "2026-09-28-14-05"


def test_falls_back_to_clock_with_three_truncated_milliseconds(metadata):
    result = RecordingTimeResolver(clock=lambda: NOW, to_local=local_time).resolve(
        Path("video.mp4")
    )
    assert result.prefix == "2026-09-28-18-04-03-007"
    assert result.source == "current_time"
    assert result.reason
    assert metadata["closed"]


def test_falls_back_on_metadata_read_error(monkeypatch):
    def fail(path):
        raise OSError("unreadable")

    monkeypatch.setattr("av.open", fail)
    result = RecordingTimeResolver(clock=lambda: NOW, to_local=local_time).resolve(
        Path("video.mp4")
    )
    assert result.prefix == "2026-09-28-18-04-03-007"
    assert "unreadable" in result.reason


def test_reads_real_container_metadata(tmp_path):
    import av
    import numpy as np

    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        container.metadata["creation_time"] = "2026-09-28T11:05:07Z"
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = RecordingTimeResolver(to_local=local_time).resolve(path)
    assert result.prefix == "2026-09-28-14-05"
    assert result.source == "container.creation_time"


@pytest.mark.parametrize(
    "offset,value,expected",
    [
        (0, "2026-09-28T11:05:07Z", "2026-09-28-11-05"),
        (3, "2026-09-28T23:30:00Z", "2026-09-29-02-30"),
        (-5, "2026-01-01T02:30:00Z", "2025-12-31-21-30"),
    ],
)
def test_uses_local_timezone_without_suffix_and_handles_date_boundary(
    metadata, offset, value, expected
):
    metadata["container"]["creation_time"] = value

    def convert(dt):
        return dt.astimezone(timezone(timedelta(hours=offset)))

    result = RecordingTimeResolver(to_local=convert).resolve(Path("video.mp4"))
    assert result.prefix == expected


def test_default_converter_uses_os_local_timezone(metadata):
    metadata["container"]["creation_time"] = "2026-09-28T11:05:07Z"
    expected = datetime(2026, 9, 28, 11, 5, 7, tzinfo=timezone.utc).astimezone()
    assert RecordingTimeResolver().resolve(Path("video.mp4")).prefix == expected.strftime(
        "%Y-%m-%d-%H-%M"
    )
    # Remove metadata so the second call exercises the fallback as well.
    metadata["container"].clear()
    result = RecordingTimeResolver(clock=lambda: NOW).resolve(Path("undated.mp4"))
    assert result.prefix == NOW.astimezone().strftime("%Y-%m-%d-%H-%M-%S") + "-007"
