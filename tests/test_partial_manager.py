"""Связь экспорта и импорта v2 с настоящей игровой реконструкцией; OCR подменён."""

import json
from copy import deepcopy

import pytest

from domino_video.corrections import Corrections
from domino_video.manager import ParseManager
from domino_video.partial_game import build_partial_game
from domino_video.reconstruct import GameReconstructor
from domino_video.video import VideoTimeline


def manager_fixture(observations, sample_game, monkeypatch):
    names = [
        p["name"]
        for p in sorted(
            [p for t in sample_game["teams"] for p in t["players"]], key=lambda p: p["seat"]
        )
    ]
    raw = GameReconstructor().extract(observations)
    for rnd, expected in zip(raw, sample_game["rounds"], strict=True):
        rnd["unresolved"] = []
        rnd["transition_windows"] = []
        rnd["events"] = [
            dict(
                seat=names.index(m["player"]),
                action=m["action"],
                stone="-".join(sorted(m["stone"].split("-"))),
                orientation=m["stone"],
                time=rnd["start"] + i,
            )
            for i, m in enumerate([m for m in expected["moves"] if m["action"] != "pass"])
        ]

    class Reader:
        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            yield from ((obs.time, obs) for obs in observations)

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    manager = ParseManager(Reader(), Recognizer())
    monkeypatch.setattr(manager._reconstructor, "extract", lambda rows: deepcopy(raw))
    # Изолируем входной неизвестный промежуток от отдельного resolver: проверяется
    # реальная build/validator после corrections, а не повторный OCR.
    monkeypatch.setattr(manager._reconstructor, "resolve_logically", lambda *args: False)
    monkeypatch.setattr(
        "domino_video.recognition_samples.RecognitionSamples.write_stones", lambda *args: False
    )
    return manager, raw, names


def test_manager_accepts_supplied_v2_and_publishes_validated_v1(
    tmp_path, observations, sample_game, monkeypatch
):
    import hashlib

    manager, raw, names = manager_fixture(observations, sample_game, monkeypatch)
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"synthetic reader fixture")
    doc = build_partial_game(
        raw, names, "withoutEggs", 50, hashlib.sha256(source.read_bytes()).hexdigest(), 1
    )
    doc["rounds"][0]["moves"][0]["recognition_status"] = "supplied"
    output = tmp_path / "out"
    assert manager.run([source], output, "withoutEggs", 50, Corrections(doc)) == 0
    result = next(output.glob("*-game-001.json"))
    assert json.loads(result.read_text(encoding="utf8")) == sample_game
    assert (tmp_path / "ready" / source.name).exists()


def test_manager_unknown_gap_roundtrip_to_full_v1(tmp_path, observations, sample_game, monkeypatch):
    manager, raw, names = manager_fixture(observations, sample_game, monkeypatch)
    raw[-1]["events"].pop()
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"synthetic reader fixture")
    output = tmp_path / "out"
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == 1
    assert source.exists()
    partial_path = next(output.glob("*-needs-review.json"))
    doc = json.loads(partial_path.read_text(encoding="utf8"))
    unknown = doc["rounds"][-1]["moves"][-1]
    assert unknown["stone"] is None
    assert unknown["metadata"]["unknown_count"] is None
    expected = [m for m in sample_game["rounds"][-1]["moves"] if m["action"] != "pass"][-1]
    unknown.update(
        recognition_status="supplied",
        player=expected["player"],
        action=expected["action"],
        stone=expected["stone"],
    )
    assert manager.run([source], output, "withoutEggs", 50, Corrections(doc)) == 0
    result = next(output.glob("*-game-001.json"))
    assert json.loads(result.read_text(encoding="utf8")) == sample_game
    assert partial_path.exists()


def test_manager_rejects_supplied_first_action_that_is_not_start(
    tmp_path, observations, sample_game, monkeypatch
):
    import hashlib

    manager, raw, names = manager_fixture(observations, sample_game, monkeypatch)
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"synthetic reader fixture")
    doc = build_partial_game(
        raw, names, "withoutEggs", 50, hashlib.sha256(source.read_bytes()).hexdigest(), 1
    )
    doc["rounds"][0]["moves"][0].update(recognition_status="supplied", action="left")
    output = tmp_path / "out"
    assert manager.run([source], output, "withoutEggs", 50, Corrections(doc)) == 1
    assert source.exists()
    assert not list(output.glob("*-game-001.json"))


def test_manager_no_move_review_of_false_gap_permits_full_legal_v1(
    tmp_path, observations, sample_game, monkeypatch
):
    manager, raw, _ = manager_fixture(observations, sample_game, monkeypatch)
    time = raw[0]["events"][0]["time"] + 0.25
    raw[0]["unresolved"] = [
        dict(
            seat=None,
            action=None,
            stone=None,
            time=time,
            interval=[time, time + 0.25],
            candidates=[],
            reason="occluded",
            metadata=dict(unknown_count=None, order_candidates=[]),
        )
    ]
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"synthetic reader fixture")
    output = tmp_path / "out"
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == 1
    path = next(output.glob("*-needs-review.json"))
    doc = json.loads(path.read_text(encoding="utf8"))
    gap = next(
        move for move in doc["rounds"][0]["moves"] if move["recognition_status"] == "unresolved"
    )
    gap.update(
        recognition_status="supplied",
        review_resolution="no_move",
        player=None,
        action="unknown",
        stone=None,
    )
    assert manager.run([source], output, "withoutEggs", 50, Corrections(doc)) == 0
    game = json.loads(next(output.glob("*-game-001.json")).read_text(encoding="utf8"))
    assert game == sample_game
    assert (tmp_path / "ready" / source.name).exists()
    assert path.exists()


@pytest.mark.parametrize("fail_partial", [False, True])
def test_partial_publication_preserves_previous_files_and_source(
    tmp_path, observations, sample_game, monkeypatch, fail_partial
):
    manager, raw, _ = manager_fixture(observations, sample_game, monkeypatch)
    raw[-1]["events"].pop()
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"synthetic reader fixture")
    output = tmp_path / "out"
    output.mkdir()
    previous_v1 = output / "2026-09-28-13-58-source-001-game-001.json"
    previous_v2 = output / "2026-09-28-13-58-source-001-game-001-needs-review.json"
    previous_v1.write_bytes(b"previous verified game")
    previous_v2.write_bytes(b"previous review document")
    write = manager._storage.write

    def publish(path, document, **kwargs):
        if fail_partial and path.name.endswith("-needs-review.json"):
            raise OSError("publication failed")
        return write(path, document, **kwargs)

    monkeypatch.setattr(manager._storage, "write", publish)
    assert manager.run([source], output, "withoutEggs", 50, Corrections()) == 1
    assert source.read_bytes() == b"synthetic reader fixture"
    assert previous_v1.read_bytes() == b"previous verified game"
    if fail_partial:
        assert previous_v2.read_bytes() == b"previous review document"
    else:
        from domino_video.partial_game import validate_partial_game

        validate_partial_game(json.loads(previous_v2.read_text(encoding="utf8")))
