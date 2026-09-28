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
