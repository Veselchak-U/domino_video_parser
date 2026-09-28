import pytest


@pytest.mark.parametrize("failure", [None, "publish", "validation"])
def test_repeated_game_export_preserves_or_replaces_result(
    tmp_path, observations, sample_game, monkeypatch, capsys, failure
):
    import json
    import os

    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager
    from domino_video.video import VideoTimeline

    class Reader:
        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            for item in observations:
                yield item.time, item

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"fixture")
    output = tmp_path / "out"
    manager = ParseManager(Reader(), Recognizer())
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == 0
    target = next(output.glob("*game*.json"))
    target.write_text('{"previous": true}', encoding="utf-8")
    previous = target.read_bytes()
    extra = output / "2026-09-28-13-58-source-001-game-002.json"
    extra.write_bytes(b"previous extra game")
    capsys.readouterr()
    if failure == "publish":
        original = os.replace

        def replace(src, dst):
            if dst == target:
                raise PermissionError("game denied")
            return original(src, dst)

        monkeypatch.setattr(os, "replace", replace)
    elif failure == "validation":
        for item in observations:
            if item.scores and item.time > 284:
                item.scores = (97, 29)
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == int(bool(failure))
    report = json.loads(next((output / "report").glob("*.json")).read_text(encoding="utf-8"))
    assert len(report["games"]) == 1
    assert report["status"] == ("needs_review" if failure else "ok")
    if failure:
        assert target.read_bytes() == previous
        assert report["games"][0]["errors"]
        if failure == "publish":
            assert "game denied" in report["games"][0]["errors"][0]["message"]
    else:
        assert json.loads(target.read_text(encoding="utf-8")) == sample_game
    assert extra.read_bytes() == b"previous extra game"
    assert not list(output.rglob(".domino-*.tmp"))
    assert ("100%" in capsys.readouterr().out) == (not failure)


def test_continues_after_missing_or_invalid_input(tmp_path):
    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    output = tmp_path / "out"
    assert (
        ParseManager().run(
            [tmp_path / "missing.mp4", broken], output, "withoutEggs", 50, Corrections()
        )
        == 1
    )
    assert len(list((output / "report").glob("*report.json"))) == 2


def test_replaces_report_on_repeated_processing(tmp_path):
    import json

    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager

    output = tmp_path / "out"
    output.mkdir()
    (output / "report").mkdir()
    existing = output / "report" / "2026-09-28-13-58-source-001-report.json"
    existing.write_text("preserved")
    assert (
        ParseManager().run(
            [tmp_path / "2026-09-28-13-58.mp4"], output, "withoutEggs", 50, Corrections()
        )
        == 1
    )
    assert json.loads(existing.read_text(encoding="utf-8"))["status"] == "error"
    assert list((output / "report").iterdir()) == [existing]


def test_exports_two_games_and_same_basename_without_collision(tmp_path, observations, sample_game):
    import json
    from copy import deepcopy

    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager

    second = deepcopy(observations)
    for o in second:
        o.time += 300
    combined = observations + second

    class Reader:
        def timeline(self, path, scanning=None):
            from domino_video.video import VideoTimeline

            return VideoTimeline(0, 600)

        def frames(self, path):
            for o in combined if path.parent.name == "first" else observations:
                yield o.time, o

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    paths = []
    for folder in ["first", "second"]:
        p = tmp_path / folder / "same_2026-09-28-13-58.mp4"
        p.parent.mkdir()
        p.write_bytes(b"fixture")
        paths.append(p)
    output = tmp_path / "out"
    assert (
        ParseManager(Reader(), Recognizer()).run(paths, output, "withoutEggs", 50, Corrections())
        == 0
    )
    files = sorted(output.glob("*game*.json"))
    assert len(files) == 3
    assert [p.name for p in files] == [
        "2026-09-28-13-58-source-001-game-001.json",
        "2026-09-28-13-58-source-001-game-002.json",
        "2026-09-28-13-58-source-002-game-001.json",
    ]
    for path in files:
        assert json.loads(path.read_text(encoding="utf-8")) == sample_game


def test_fallback_time_shared_by_all_outputs_and_ambiguous_source_continues(tmp_path, observations):
    import json
    from copy import deepcopy
    from datetime import datetime, timedelta, timezone

    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager
    from domino_video.recording_time import RecordingTimeResolver

    later = deepcopy(observations)
    for o in later:
        o.time += 300

    class Reader:
        def timeline(self, path, scanning=None):
            from domino_video.video import VideoTimeline

            return VideoTimeline(0, 600)

        def frames(self, path):
            for o in observations + later:
                yield o.time, o

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    calls = []

    def clock():
        calls.append(1)
        return datetime(2026, 9, 28, 15, 4, 3, 7999, tzinfo=timezone.utc)

    ambiguous = tmp_path / "2026-09-28-13-58_2026-09-29-13-58.mp4"
    valid = tmp_path / "video.mp4"
    for p in [ambiguous, valid]:
        p.write_bytes(b"fixture")
    output = tmp_path / "out"
    code = ParseManager(
        Reader(),
        Recognizer(),
        RecordingTimeResolver(
            clock=clock, to_local=lambda value: value.astimezone(timezone(timedelta(hours=3)))
        ),
    ).run([ambiguous, valid], output, "withoutEggs", 50, Corrections())
    assert code == 1
    assert calls == [1]
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*.json")) == [
        "2026-09-28-18-04-03-007-source-002-game-001.json",
        "2026-09-28-18-04-03-007-source-002-game-002.json",
        "report/2026-09-28-18-04-03-007-source-002-report.json",
        "report/undated-source-001-report.json",
    ]
    report = json.loads(
        (output / "report" / "2026-09-28-18-04-03-007-source-002-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["recording_time"]["source"] == "current_time"
    assert report["recording_time"]["reason"]
    assert report["status"] == "ok"


def test_rejects_conflicting_score(tmp_path, observations):
    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager

    for o in observations:
        if o.scores and o.time > 284:
            o.scores = (97, 29)

    class Reader:
        def timeline(self, path, scanning=None):
            from domino_video.video import VideoTimeline

            return VideoTimeline(0, 600)

        def frames(self, path):
            for o in observations:
                yield o.time, o

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    p = tmp_path / "video.mp4"
    p.write_bytes(b"fixture")
    assert (
        ParseManager(Reader(), Recognizer()).run(
            [p], tmp_path / "out", "withoutEggs", 50, Corrections()
        )
        == 1
    )
    assert not list((tmp_path / "out").glob("*game*.json"))


def test_name_correction_restores_export_and_bad_input_does_not_stop_it(
    tmp_path, observations, sample_game
):
    import hashlib
    import json

    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager

    for o in observations:
        o.names = None

    class Reader:
        def timeline(self, path, scanning=None):
            from domino_video.video import VideoTimeline

            return VideoTimeline(0, 600)

        def frames(self, path):
            for o in observations:
                yield o.time, o

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    source = tmp_path / "valid.mp4"
    source.write_bytes(b"fixture")
    fix = Corrections(
        {
            "sources": [
                {
                    "sha256": hashlib.sha256(b"fixture").hexdigest(),
                    "games": [{"number": 1, "teams": sample_game["teams"], "rounds": []}],
                }
            ]
        }
    )
    output = tmp_path / "out"
    code = ParseManager(Reader(), Recognizer()).run(
        [tmp_path / "missing.mp4", source], output, "withoutEggs", 50, fix
    )
    assert code == 1
    game = next(output.glob("*game*.json"))
    assert json.loads(game.read_text(encoding="utf-8")) == sample_game


def test_report_write_failure_does_not_show_success(tmp_path, observations, monkeypatch, capsys):
    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager
    from domino_video.video import VideoTimeline

    class Reader:
        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            for item in observations:
                yield item.time, item

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    manager = ParseManager(Reader(), Recognizer())
    original = manager._storage.write

    def write(path, value, **kwargs):
        if path.name.endswith("-report.json"):
            raise OSError("report denied")
        original(path, value, **kwargs)

    monkeypatch.setattr(manager._storage, "write", write)
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"fixture")
    assert manager.run([source], tmp_path / "out", "withoutEggs", 50, Corrections()) == 1
    text = capsys.readouterr().out
    assert "100%" not in text
    assert "report denied" in text


def test_interrupt_finishes_progress_and_does_not_start_next_source(tmp_path, capsys):
    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager
    from domino_video.video import VideoTimeline

    calls = []

    class Reader:
        def timeline(self, path, scanning=None):
            calls.append(path)
            return VideoTimeline(0, 10)

        def frames(self, path):
            raise KeyboardInterrupt
            yield

    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"fixture")
    assert (
        ParseManager(reader=Reader()).run(
            [source, source], tmp_path / "out", "withoutEggs", 50, Corrections()
        )
        == 1
    )
    assert calls == [source]
    text = capsys.readouterr().out
    assert "прервана" in text
    assert "100%" not in text


def test_keeps_old_root_report_and_writes_new_one_in_subdirectory(tmp_path):
    import json

    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager

    output = tmp_path / "custom output"
    output.mkdir()
    name = "2026-09-28-13-58-source-001-report.json"
    old = output / name
    old.write_text("old report")
    assert (
        ParseManager().run(
            [tmp_path / "2026-09-28-13-58.mp4"], output, "withoutEggs", 50, Corrections()
        )
        == 1
    )
    assert old.read_text() == "old report"
    assert json.loads((output / "report" / name).read_text(encoding="utf-8"))["status"] == "error"


def test_file_in_place_of_report_directory_produces_clear_failure(tmp_path, capsys):
    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager

    output = tmp_path / "out"
    output.mkdir()
    obstacle = output / "report"
    obstacle.write_text("preserve")
    assert (
        ParseManager().run(
            [tmp_path / "2026-09-28-13-58.mp4"], output, "withoutEggs", 50, Corrections()
        )
        == 1
    )
    text = capsys.readouterr().out
    assert "Не удалось сохранить отчёт" in text
    assert "100%" not in text
    assert obstacle.read_text() == "preserve"
    assert list(output.iterdir()) == [obstacle]
