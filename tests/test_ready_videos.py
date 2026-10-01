import json

import pytest

from domino_video.corrections import Corrections
from domino_video.manager import ParseManager
from domino_video.progress import ConsoleProgress
from domino_video.video import VideoTimeline


@pytest.fixture
def processing(tmp_path, observations):
    class Reader:
        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            for item in observations:
                yield item.time, item

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    source = tmp_path / "in" / "Игра 2026-09-28-13-58.MP4"
    source.parent.mkdir()
    source.write_bytes(b"video fixture")
    return ParseManager(Reader(), Recognizer()), source, tmp_path / "Мои результаты"


@pytest.mark.parametrize("custom_output", [False, True])
def test_moves_only_after_all_games_and_report_are_saved(
    processing, observations, monkeypatch, capsys, custom_output
):
    from copy import deepcopy

    from domino_video.storage import ExportStorage

    manager, source, output = processing
    if not custom_output:
        output = source.parent.parent / "out"
    second = deepcopy(observations)
    for item in second:
        item.time += 300
    observations.extend(second)
    old_ready = output / "ready"
    old_ready.mkdir(parents=True)
    old_video = old_ready / source.name
    old_video.write_bytes(b"previous output video")
    original = ExportStorage.move_video

    def move(path, ready):
        report = json.loads(next((output / "report").glob("*.json")).read_text("utf-8"))
        assert report["status"] == "ok"
        assert report["source"] == str(source.resolve())
        assert len(list(output.glob("*game*.json"))) == len(report["games"]) == 2
        assert source.exists()
        assert "100%" not in capsys.readouterr().out
        return original(manager._storage, path, ready)

    monkeypatch.setattr(manager._storage, "move_video", move)
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == 0
    assert not source.exists()
    target = source.parent / "ready" / source.name
    assert target.read_bytes() == b"video fixture"
    assert old_video.read_bytes() == b"previous output video"
    text = capsys.readouterr().out
    assert str(target) in text
    assert "100%" in text


@pytest.mark.parametrize("failure", ["game", "sample", "stone_sample", "report"])
def test_keeps_source_on_processing_or_report_failure(processing, monkeypatch, capsys, failure):
    manager, source, output = processing
    original = manager._storage.write

    def write(path, value, **kwargs):
        if (failure == "game" and "-game-" in path.name) or (
            failure == "report" and "-report" in path.name
        ):
            raise OSError("write denied")
        return original(path, value, **kwargs)

    monkeypatch.setattr(manager._storage, "write", write)
    if failure == "sample":
        monkeypatch.setattr(manager._samples, "write", lambda *args: True)
    if failure == "stone_sample":
        monkeypatch.setattr(manager._samples, "write_stones", lambda *args: True)
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == 1
    assert source.read_bytes() == b"video fixture"
    assert not (source.parent / "ready").exists()
    assert "100%" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["move", "interrupt", "directory"])
def test_move_failure_preserves_results_and_handles_next_source(
    processing, monkeypatch, capsys, failure
):
    from domino_video.storage import ExportStorage

    manager, source, output = processing
    second = source.with_name("next 2026-09-28-13-58.mp4")
    second.write_bytes(b"next video")
    original = ExportStorage.move_video

    def move(path, ready):
        if path == source:
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise PermissionError("move denied")
        return original(manager._storage, path, ready)

    if failure == "directory":
        (source.parent / "ready").write_bytes(b"obstacle")
    else:
        monkeypatch.setattr(manager._storage, "move_video", move)
    assert manager.run([source, second], output, "withoutEggs", 50, Corrections()) == 1
    assert source.read_bytes() == b"video fixture"
    text = capsys.readouterr().out
    if failure == "move":
        assert not second.exists()
        assert (source.parent / "ready" / second.name).read_bytes() == b"next video"
        first, following = text.split("[2/2]")
        assert "100%" not in first
        assert "100%" in following
        assert "Не удалось переместить видео" in first
    else:
        assert second.exists()
        assert "100%" not in text
        if failure == "interrupt":
            assert "[2/2]" not in text
            assert "прервана" in text
        else:
            assert (source.parent / "ready").read_bytes() == b"obstacle"
    report = json.loads(next((output / "report").glob("*.json")).read_text("utf-8"))
    assert report["status"] == "ok"
    assert list(output.glob("*game*.json"))


def test_successful_source_moves_after_previous_source_error(processing):
    manager, source, output = processing
    assert (
        manager.run(
            [source.with_name("missing.mp4"), source], output, "withoutEggs", 50, Corrections()
        )
        == 1
    )
    assert not source.exists()
    assert (source.parent / "ready" / source.name).read_bytes() == b"video fixture"


@pytest.mark.parametrize("ready_name", ["ready", "READY"])
def test_successful_reprocessing_in_ready_keeps_same_video(processing, ready_name):
    manager, source, output = processing
    ready = source.parent / ready_name
    ready.mkdir(parents=True)
    target = ready / source.name
    source.rename(target)
    assert manager.run([target], output, "withoutEggs", 50, Corrections()) == 0
    assert list(ready.iterdir()) == [target]
    assert target.read_bytes() == b"video fixture"


@pytest.mark.parametrize("custom_input", [False, True])
def test_cli_moves_video_into_selected_input_ready(processing, monkeypatch, custom_input):
    from domino_video.cli import main

    manager, source, output = processing
    monkeypatch.chdir(source.parent.parent)
    monkeypatch.setattr("domino_video.manager.ParseManager", lambda: manager)
    args = [
        "--output",
        str(output),
        "--fish-variant",
        "withoutEggs",
        "--score-limit",
        "50",
        "--workers",
        "1",
        "--device",
        "cpu",
    ]
    if custom_input:
        folder = source.parent.parent / "Мои записи"
        folder.mkdir()
        target = folder / source.name
        source.rename(target)
        source = target
        args.extend(["--input", str(folder)])
    assert main(args) == 0
    assert not source.exists()
    assert (source.parent / "ready" / source.name).read_bytes() == b"video fixture"
    assert not (output / "ready").exists()
    assert main(args) == 1


def test_final_time_and_speed_include_move(processing, monkeypatch, capsys):
    from domino_video.storage import ExportStorage

    manager, source, output = processing
    now = [0.0]
    monkeypatch.setattr(
        "domino_video.manager.ConsoleProgress", lambda: ConsoleProgress(clock=lambda: now[0])
    )
    original_write = manager._storage.write
    original_move = ExportStorage.move_video

    def write(path, value, **kwargs):
        result = original_write(path, value, **kwargs)
        if "-report" in path.name:
            now[0] = 10.0
        return result

    def move(path, ready):
        assert "100%" not in capsys.readouterr().out
        now[0] += 5.0
        return original_move(manager._storage, path, ready)

    monkeypatch.setattr(manager._storage, "write", write)
    monkeypatch.setattr(manager._storage, "move_video", move)
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == 0
    assert "100% за 15 сек скорость 19.4x" in capsys.readouterr().out
