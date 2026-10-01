import json

import pytest


def test_never_replaces_existing_result(tmp_path):
    from domino_video.storage import ExportStorage

    target = tmp_path / "game.json"
    ExportStorage().write(target, {"name": "Игрок"})
    with pytest.raises(FileExistsError):
        ExportStorage().write(target, {"name": "Другой"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"name": "Игрок"}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("collision", [False, True])
def test_moves_video_without_overwriting_existing_paths(tmp_path, collision):
    from domino_video.storage import ExportStorage

    source = tmp_path / "Запись игры.MP4"
    content = bytes(range(256)) * 8193
    source.write_bytes(content)
    ready = tmp_path / "Мои результаты" / "ready"
    if collision:
        ready.mkdir(parents=True)
        (ready / source.name).write_bytes(b"old video")
        (ready / f"{source.stem}-001{source.suffix}").mkdir()
    target = ExportStorage().move_video(source, ready)
    assert target == ready / (f"{source.stem}-002{source.suffix}" if collision else source.name)
    assert target.read_bytes() == content
    assert not source.exists()
    if collision:
        assert (ready / source.name).read_bytes() == b"old video"
        assert (ready / f"{source.stem}-001{source.suffix}").is_dir()


def test_keeps_video_already_in_ready(tmp_path):
    from domino_video.storage import ExportStorage

    ready = tmp_path / "ready"
    ready.mkdir()
    source = ready / "game.mp4"
    source.write_bytes(b"video")
    assert ExportStorage().move_video(source, ready / ".." / "ready") == source
    assert list(ready.iterdir()) == [source]
    assert source.read_bytes() == b"video"


@pytest.mark.parametrize("failure", ["copy", "sync", "unlink", "interrupt"])
def test_failed_move_preserves_source_and_cleans_its_copy(tmp_path, monkeypatch, failure):
    import os
    import shutil
    from pathlib import Path

    from domino_video.storage import ExportStorage

    source = tmp_path / "game.mp4"
    source.write_bytes(b"complete video")
    ready = tmp_path / "ready"
    ready.mkdir()
    old = ready / source.name
    old.write_bytes(b"previous")

    def copy(src, dst, *args, **kwargs):
        dst.write(b"partial")
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise OSError("copy denied")

    def sync(fd):
        raise OSError("sync denied")

    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == source:
            assert (ready / "game-001.mp4").read_bytes() == source.read_bytes()
            raise PermissionError("unlink denied")
        return original_unlink(path, *args, **kwargs)

    if failure in ("copy", "interrupt"):
        monkeypatch.setattr(shutil, "copyfileobj", copy)
    elif failure == "sync":
        monkeypatch.setattr(os, "fsync", sync)
    else:
        monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
        ExportStorage().move_video(source, ready)
    assert source.read_bytes() == b"complete video"
    assert old.read_bytes() == b"previous"
    assert list(ready.iterdir()) == [old]


def test_move_does_not_require_same_filesystem(tmp_path, monkeypatch):
    from pathlib import Path

    from domino_video.storage import ExportStorage

    def rename(*args, **kwargs):
        raise OSError(18, "cross-device link")

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(Path, "replace", rename)
    source = tmp_path / "game.mp4"
    source.write_bytes(b"video")
    assert ExportStorage().move_video(source, tmp_path / "ready").read_bytes() == b"video"
    assert not source.exists()


def test_rejects_duplicate_keys():
    from domino_video.storage import read_json

    with pytest.raises(ValueError, match="Повтор"):
        read_json('{"x":1,"x":2}')


def test_removes_temporary_file_on_publish_failure(tmp_path, monkeypatch):
    import os

    from domino_video.storage import ExportStorage

    def fail(*args):
        raise PermissionError("Cannot publish")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(PermissionError):
        ExportStorage().write(tmp_path / "game.json", {})
    assert not list(tmp_path.iterdir())


def test_replaces_report_atomically_and_cleans_temp_on_failure(tmp_path, monkeypatch):
    import os

    from domino_video.storage import ExportStorage

    target = tmp_path / "report.json"
    storage = ExportStorage()
    storage.write(target, {"status": "old"})
    storage.write(target, {"status": "new"}, replace=True)
    assert json.loads(target.read_text()) == {"status": "new"}

    def fail(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(PermissionError):
        storage.write(target, {"status": "lost"}, replace=True)
    assert json.loads(target.read_text()) == {"status": "new"}
    assert list(tmp_path.iterdir()) == [target]
