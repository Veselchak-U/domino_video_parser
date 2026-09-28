import subprocess
import sys
from pathlib import Path

import pytest

from domino_video.cli import main

RULES = ["--fish-variant", "withoutEggs", "--score-limit", "50"]


def test_requires_explicit_game_parameters():
    result = subprocess.run(
        [sys.executable, "-m", "domino_video", "video.mp4"],
        capture_output=True,
    )
    assert result.returncode == 2
    assert "обязательн" in result.stderr.decode("utf-8").lower()


@pytest.mark.parametrize("custom_input", [False, True])
@pytest.mark.parametrize("custom_output", [False, True])
def test_selects_directory_defaults_and_overrides(
    tmp_path, monkeypatch, custom_input, custom_output
):
    monkeypatch.chdir(tmp_path)
    source = Path("входные видео" if custom_input else "in")
    target = Path("мои результаты" if custom_output else "out")
    source.mkdir()
    for name in ["b.MP4", "a.mov", "notes.txt"]:
        (source / name).touch()
    (source / "nested.mp4").mkdir()
    (source / "nested.mp4" / "hidden.mp4").touch()
    calls = []
    monkeypatch.setattr(
        "domino_video.manager.ParseManager.run", lambda self, *args: calls.append(args) or 0
    )
    argv = RULES + (["--input", str(source)] if custom_input else [])
    argv += ["--output", str(target)] if custom_output else []
    assert main(argv) == 0
    assert calls[0][:4] == ([source / "a.mov", source / "b.MP4"], target, "withoutEggs", 50)


def test_selects_all_supported_extensions(tmp_path, monkeypatch):
    names = [
        f"{i:02d}.{ext}"
        for i, ext in enumerate(
            ["MP4", "mkv", "MOV", "avi", "webm", "m4v", "mpeg", "mpg", "ts", "mts", "m2ts"]
        )
    ]
    for name in reversed(names):
        (tmp_path / name).touch()
    calls = []
    monkeypatch.setattr(
        "domino_video.manager.ParseManager.run", lambda self, *args: calls.append(args) or 0
    )
    assert main(RULES + ["--input", str(tmp_path)]) == 0
    assert calls[0][0] == [tmp_path / name for name in names]


@pytest.mark.parametrize("kind", ["missing", "file", "empty", "text_only", "denied"])
def test_rejects_unusable_directory(tmp_path, monkeypatch, capsys, kind):
    source = tmp_path / "input"
    if kind == "file":
        source.touch()
    elif kind != "missing":
        source.mkdir()
    if kind == "text_only":
        (source / "notes.txt").touch()
    if kind == "denied":

        def denied(self):
            raise PermissionError("access denied")

        monkeypatch.setattr(Path, "iterdir", denied)

    def unexpected(*args):
        pytest.fail("Recognition must not start")

    monkeypatch.setattr("domino_video.manager.ParseManager.run", unexpected)
    output = tmp_path / "out"
    assert main(RULES + ["--input", str(source), "--output", str(output)]) == 1
    assert "Ошибка" in capsys.readouterr().err
    assert not output.exists()
    if kind == "missing":
        assert not source.exists()


@pytest.mark.parametrize("extra", [[], ["--input", "in"]])
def test_rejects_positional_videos(extra):
    with pytest.raises(SystemExit) as error:
        main(["video.mp4", "second.mp4", "--output", "out"] + RULES + extra)
    assert error.value.code == 2


@pytest.mark.parametrize("rules", [[], RULES[:2], RULES[2:]])
def test_directory_mode_requires_both_rules(rules):
    with pytest.raises(SystemExit) as error:
        main(["--input", "in"] + rules)
    assert error.value.code == 2


def test_directory_exports_valid_game_after_corrupt_video(
    tmp_path, monkeypatch, observations, sample_game
):
    import json

    from domino_video.manager import ParseManager

    class Reader:
        def frames(self, path):
            if path.name == "a.mp4":
                raise ValueError("Повреждённое видео")
            for observation in observations:
                yield observation.time, observation

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    manager = ParseManager(Reader(), Recognizer())
    monkeypatch.setattr("domino_video.manager.ParseManager", lambda: manager)
    monkeypatch.chdir(tmp_path)
    Path("in").mkdir()
    for name in ["a.mp4", "b.mp4"]:
        (Path("in") / name).write_bytes(b"fixture")
    assert main(RULES) == 1
    game = next(Path("out").glob("*-source-002-game-001.json"))
    assert json.loads(game.read_text(encoding="utf-8")) == sample_game
    assert len(list(Path("out").glob("*-report.json"))) == 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows BAT")
def test_bat_preserves_working_directory_and_exit_codes(tmp_path):
    import json

    bat = Path(__file__).resolve().parents[1] / "tools" / "run.bat"
    source = tmp_path / "in"
    source.mkdir()
    (source / "тест видео.mp4").write_bytes(b"broken video")
    result = subprocess.run([str(bat), *RULES], cwd=tmp_path, capture_output=True)
    assert result.returncode == 1, result.stderr
    report = next((tmp_path / "out").glob("*-source-001-report.json"))
    assert json.loads(report.read_text(encoding="utf-8"))["source"] == str(
        source / "тест видео.mp4"
    )
    result = subprocess.run([str(bat), "--help"], cwd=tmp_path, capture_output=True)
    assert result.returncode == 0
    assert b"--input" in result.stdout
    result = subprocess.run([str(bat), "video.mp4", *RULES], cwd=tmp_path, capture_output=True)
    assert result.returncode == 2
