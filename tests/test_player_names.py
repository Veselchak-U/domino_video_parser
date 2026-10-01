import json
import string

import cv2
import pytest

from domino_video.corrections import Corrections
from domino_video.manager import ParseManager
from domino_video.video import VideoTimeline
from domino_video.vision import Observation


def attempt(text, confidences=None, seat=1, time=20):
    confidences = confidences or [0.99] * len(text)
    return dict(
        field="name",
        seat=seat,
        team=None,
        time=time,
        raw_rows=[dict(text=text, confidence=sum(confidences) / len(text))] if text else [],
        symbols=[
            dict(text=c, confidence=p, start=i, end=i + 1)
            for i, (c, p) in enumerate(zip(text, confidences))
        ],
        accepted_text=text,
        threshold=0.7,
        reason=None,
    )


def resolve(attempts, start=20, end=100):
    from domino_video.player_names import PlayerNameResolver

    observations = [
        Observation(a["time"], [], [[], [], [], []], None, False, True, ocr_attempts=[a])
        for a in attempts
    ]
    return PlayerNameResolver().resolve(observations, [dict(start=start, end=end)])


def test_partial_name_and_exact_threshold():
    result = resolve([attempt("Alex@7", [0.99, 0.99, 0.7, 0.2, 0.99, 0.99])])
    assert result.names == ["Al**@7", "Unrecognized_1", "Unrecognized_2", "Unrecognized_3"]
    assert result.entries[0]["masked_positions"] == [2, 3]
    assert result.entries[0]["reason"] == "partial_name"


@pytest.mark.parametrize(
    "text",
    [
        "МиХ@ЛыЧ",
        "Ёж-42!",
        "Игрок№1",
        "Alex_Игрок",
        string.punctuation,
        "@!",
        "*",
        "علي",
        "राज",
        "דוד",
        "e\u0301",
        "Player_69271431",
        " A B ",
    ],
)
def test_confident_unicode_and_keyboard_symbols_survive(text):
    result = resolve([attempt(text)])
    assert result.names[0] == text
    assert not any(e["seat"] == 1 for e in result.entries)


def test_unknown_numbers_reserve_real_names_and_reset_each_game():
    attempts = [attempt("علي", [0.7] * 3), attempt("Unrecognized_1", seat=3)]
    result = resolve(attempts)
    assert result.names == ["Unrecognized_2", "Unrecognized_3", "Unrecognized_1", "Unrecognized_4"]
    assert result.entries[0]["reason"] == "generated_name"
    assert result.entries[1]["reason"] == "no_frame"
    assert resolve([]).names == [f"Unrecognized_{i}" for i in range(1, 5)]


def test_duplicate_suffix_reserves_existing_originals():
    result = resolve(
        [attempt(n, seat=i) for i, n in enumerate(["А***", "А***", "А***_2", "Unrecognized_1"], 1)]
    )
    assert result.names == ["А***", "А***_3", "А***_2", "Unrecognized_1"]
    assert result.entries == [dict(result.entries[0])]
    assert result.entries[0]["reason"] == "duplicate_name"


def test_candidate_order_and_independent_seats_with_interval_edges():
    attempts = [
        attempt("Outside", time=9),
        attempt("Outside", time=101),
        attempt("AX", [0.99, 0.1], time=10),
        attempt("A", [0.9], time=11),
        attempt("B", [0.95], time=13),
        attempt("C", [0.95], time=12),
        attempt("Second", seat=2, time=100),
        attempt("Long?", [0.9] * 4 + [0.1], seat=3, time=20),
        attempt("Two", seat=3, time=21),
    ]
    assert resolve(attempts).names[:3] == ["C", "Second", "Long*"]


def test_candidate_mean_includes_confident_spaces_but_count_does_not():
    assert resolve([attempt("A ", [0.99, 0.81]), attempt("B", [0.95])]).names[0] == "B"


@pytest.mark.parametrize("missing", [False, True])
def test_unreadable_name_exports_valid_game_with_warning_and_sample(
    tmp_path,
    observations,
    missing,
):
    for obs in observations:
        obs.names = None
        obs.ocr_attempts = []
    if not missing:
        next(o for o in observations if 15 < o.time < 25).ocr_attempts = [
            attempt("Alex@7", [0.99, 0.99, 0.7, 0.2, 0.99, 0.99])
        ]

    class Reader:
        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            for obs in observations:
                yield obs.time, obs

        def frame_at(self, path, time):
            return cv2.imread("tests/fixtures/frame_20.png")

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"fixture")
    output = tmp_path / "out"
    assert (
        ParseManager(Reader(), Recognizer()).run([source], output, "withoutEggs", 50, Corrections())
        == 0
    )
    report = json.loads(next((output / "report").glob("*.json")).read_text(encoding="utf-8"))
    game = json.loads(next(output.glob("*game*.json")).read_text(encoding="utf-8"))
    names = [p["name"] for t in game["teams"] for p in t["players"]]
    assert len(set(names)) == 4
    entries = report["games"][0]["recognition_diagnostics"]
    assert all(e["severity"] == "warning" for e in entries)
    assert bool(entries[0]["sample"]) == (not missing)
    for rnd in game["rounds"]:
        assert set(rnd["deal"]) == set(names)


@pytest.mark.parametrize("damage", ["score", "manual_duplicate"])
def test_unknown_names_do_not_hide_game_errors_or_repair_invalid_corrections(
    tmp_path,
    observations,
    sample_game,
    damage,
):
    import hashlib
    from copy import deepcopy

    from test_recognition_diagnostics import run_fixture

    for obs in observations:
        obs.names = None
        obs.ocr_attempts = []
    corrections = Corrections()
    if damage == "score":
        for obs in observations:
            if obs.scores and obs.time > 284:
                obs.scores = (97, 29)
    else:
        teams = deepcopy(sample_game["teams"])
        teams[1]["players"][0]["name"] = teams[0]["players"][0]["name"]
        corrections = Corrections(
            dict(
                sources=[
                    dict(
                        sha256=hashlib.sha256(b"fixture").hexdigest(),
                        games=[dict(number=1, teams=teams, rounds=[])],
                    )
                ]
            )
        )
    code, report, output, _ = run_fixture(tmp_path, observations, corrections=corrections)
    assert code == 1 and report["status"] == "needs_review"
    assert not list(output.glob("*game*.json"))
    errors = report["games"][0]["errors"]
    assert any(
        "табло" in e["message"] if damage == "score" else "различ" in e["message"] for e in errors
    )


def test_player_names_change_between_games_without_sharing_candidates():
    attempts = [attempt("First", time=20), attempt("Second", time=200)]
    assert resolve(attempts, start=20, end=100).names[0] == "First"
    assert resolve(attempts, start=200, end=300).names[0] == "Second"


@pytest.mark.parametrize("fault", ["недоступна", "повреждена"])
def test_required_model_failure_is_an_initialization_error_without_generated_names(
    tmp_path,
    observations,
    monkeypatch,
    fault,
):
    from test_recognition_diagnostics import run_fixture

    from domino_video.name_ocr import NameOCR

    def fail(*args):
        raise RuntimeError(f"Локальная модель имён {fault}")

    monkeypatch.setattr(NameOCR, "model_path", fail)
    code, report, output, _ = run_fixture(tmp_path, observations)
    assert code == 1 and report["status"] == "error"
    assert report["games"] == []
    assert fault in report["errors"][0]["message"]
    assert not list(output.glob("*game*.json"))
