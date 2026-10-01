import hashlib
from types import SimpleNamespace

import cv2
import pytest

from domino_video.recognition_samples import RecognitionSamples
from domino_video.storage import ExportStorage
from domino_video.video import VideoReader


@pytest.mark.parametrize("fail", [False, True])
def test_stone_sample_published_once_or_reports_error(tmp_path, monkeypatch, fail):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    digest = hashlib.sha256(b"source").hexdigest()
    calls = []

    def frame(path, t):
        calls.append(t)
        return cv2.imread("tests/fixtures/frame_20.png")

    reader = SimpleNamespace(frame_at=frame)
    storage = ExportStorage()
    if fail:

        def broken(*a):
            raise OSError("disk full")

        monkeypatch.setattr(storage, "write_sample", broken)
    entries = [
        dict(method="late_reading", time=20, region=[700, 240, 160, 160], sample=None),
        dict(method="exclusion", time=None, region=None, sample=None),
    ]
    samples = RecognitionSamples(reader, storage)
    assert samples.write_stones(source, digest, tmp_path, entries) is fail
    if fail:
        assert entries[0]["sample"] is None
        assert "disk full" in entries[0]["sample_error"]
    else:
        assert entries[0]["sample"]["time"] == 20
        before = (tmp_path / entries[0]["sample"]["path"]).read_bytes()
        samples.write_stones(source, digest, tmp_path, entries)
        assert (tmp_path / entries[0]["sample"]["path"]).read_bytes() == before
        assert len(list((tmp_path / "samples").glob("*.png"))) == 1
    assert entries[1]["sample"] is None


def test_changed_video_does_not_publish_wrong_sample(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"changed")
    samples = RecognitionSamples(SimpleNamespace(), ExportStorage())
    entry = dict(method="animation", time=20, region=[700, 240, 160, 160], sample=None)
    assert samples.write_stones(source, "oldhash", tmp_path, [entry])
    assert entry["sample"] is None and "изменилось" in entry["sample_error"]


@pytest.mark.parametrize("early_close", [False, True])
def test_interval_decoder_merges_windows_preserves_pts_and_closes(monkeypatch, early_close):
    from test_video import Container

    container = Container(times=(1, 1.1, 1.2, 1.3, 1.4, 2, 2.1))
    seeks = []
    container.seek = lambda offset, **k: seeks.append(offset)
    container.decode = lambda **k: (
        SimpleNamespace(time=t, to_ndarray=lambda format: t) for t in container.times
    )
    monkeypatch.setattr("domino_video.video.av.open", lambda p: container)
    frames = VideoReader().interval_frames("video.mp4", [(1.1, 1.2), (1.2, 1.3), (2, 2.1)])
    if early_close:
        assert next(frames)[0] == 1.1
        frames.close()
    else:
        assert [t for t, _ in frames] == [1.1, 1.2, 1.3, 2, 2.1]
        assert seeks == [1100000, 2000000]
    assert container.closed
