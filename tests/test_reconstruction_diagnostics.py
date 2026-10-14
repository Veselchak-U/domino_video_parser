import gzip
import json
from copy import deepcopy

import pytest

from domino_video.partial_game import validate_partial_game
from domino_video.reconstruct import GameReconstructor, ReconstructionError


def real_rounds():
    with gzip.open("tests/fixtures/reconstruction_failures.json.gz", "rt", encoding="utf8") as f:
        return json.load(f)["rounds"]


def test_real_failure_has_first_dead_event_and_start_conflicts():
    with pytest.raises(ReconstructionError) as caught:
        GameReconstructor().build(real_rounds(), list("ABCD"), "withoutEggs", 50)
    diagnostic = caught.value.diagnostic
    assert diagnostic["code"] == "no_consistent_sequence"
    assert (diagnostic["round"], diagnostic["event"]) == (1, 3)
    assert diagnostic["time"] == pytest.approx(22.44678888888889)
    failure = diagnostic["details"]["first_failure"]
    assert failure["stone"] == "4-6"
    assert failure["states_before"] > 0 and failure["states_after"] == 0
    assert set(failure["rejections"]) == {"initial_hand_conflict", "endpoint_mismatch"}
    assert diagnostic["details"]["events"][1]["warnings"][0]["code"] == "start_position_conflict"
    assert "событие 3" in str(caught.value)
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize(
    "kind,code",
    [
        ("incomplete", "incomplete_round"),
        ("start", "missing_start"),
        ("duplicate", "duplicate_stones"),
        ("remaining", "unconfirmed_remaining"),
        ("missing", "missing_stones"),
        ("issues", "unresolved_events"),
    ],
)
def test_preflight_failures_have_distinct_codes_and_evidence(kind, code):
    rounds = real_rounds()
    rnd = rounds[0]
    if kind == "incomplete":
        rnd["complete"] = False
    elif kind == "start":
        rnd["start_observed"] = False
    elif kind == "duplicate":
        rnd["events"].append(deepcopy(rnd["events"][0]))
    elif kind in ("remaining", "missing"):
        rnd["events"] = rnd["events"][:-2]
        if kind == "remaining":
            rnd["remaining_confirmed"][2] = False
    else:
        rnd["issues"] = ["conflicting readings"]
    with pytest.raises(ReconstructionError) as caught:
        GameReconstructor().build(rounds, list("ABCD"), "withoutEggs", 50)
    diagnostic = caught.value.diagnostic
    assert diagnostic["code"] == code
    assert diagnostic["round"] == 1
    if kind == "remaining":
        assert diagnostic["details"]["unconfirmed_seats"] == [3]
        assert diagnostic["details"]["missing"]
        assert diagnostic["details"]["remaining"][2] == []


def test_branch_examples_are_bounded_without_losing_counts():
    rnd = dict(remaining=[[], [], [], []], initial_hand=["1-1"], end=5)
    event = dict(time=1, stone="1-1", seat=1, seats=[1] * 20, action="start")
    trace = []
    assert GameReconstructor()._round_options(rnd, [event], list("ABCD"), 1, trace) == []
    reason = trace[0]["rejections"]["initial_hand_conflict"]
    assert reason["count"] == 20
    assert len(reason["examples"]) == 2
    assert reason["omitted_examples"] == 18


def test_diagnostics_do_not_change_valid_game(observations, sample_game):
    r = GameReconstructor()
    rounds = r.extract(observations)
    before = deepcopy(rounds)
    names = next(o.names for o in observations if o.names)
    assert r.build(rounds, names, "withoutEggs", 50) == sample_game
    for original, enriched in zip(before, rounds, strict=True):
        for field in (
            "start",
            "end",
            "complete",
            "initial_hand",
            "remaining",
            "remaining_confirmed",
        ):
            assert original.get(field) == enriched.get(field)
        by_stone = {event["stone"]: event for event in enriched["events"]}
        assert len(by_stone) == len(enriched["events"])
        for event in original["events"]:
            confirmed = by_stone[event["stone"]]
            for field in ("time", "confirmed_at", "positions", "tile_size"):
                assert event.get(field) == confirmed.get(field)
            assert confirmed["seat"] in event.get("seats", [event["seat"]])
            assert confirmed["action"] in ("start", "left", "right")
    assert r.build(deepcopy(rounds), names, "withoutEggs", 50) == sample_game


@pytest.mark.parametrize("timestamp", [22.44678888888889, None])
def test_manager_publishes_structured_failure_without_inventing_time(
    tmp_path, observations, monkeypatch, timestamp
):
    from test_recognition_diagnostics import run_fixture

    def fail(self, rounds, names, variant, limit):
        raise ReconstructionError(
            "Кон 1: диагностический отказ",
            code="no_consistent_sequence",
            round_number=1,
            event=3,
            time=timestamp,
            details={"stone": "4-6"},
        )

    monkeypatch.setattr(GameReconstructor, "build", fail)
    code, report, output, _ = run_fixture(tmp_path, observations)
    assert code == 1 and report["status"] == "needs_review"
    game = report["games"][0]
    assert game["errors"][0]["time"] == timestamp
    assert game["errors"][0]["code"] == "no_consistent_sequence"
    assert game["reconstruction_diagnostics"][0]["event"] == 3
    assert game["reconstruction_diagnostics"][0]["game"] == 1
    documents = [
        json.loads(path.read_text(encoding="utf-8")) for path in output.glob("*-game-*.json")
    ]
    assert not any(document["format"] == "kozel-game-v1" for document in documents)
    assert len(documents) == 1
    validate_partial_game(documents[0])
    assert documents[0]["status"] == "needs_review"
    assert (tmp_path / "2026-09-28-13-58.mp4").exists()
