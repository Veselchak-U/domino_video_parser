import json
from copy import deepcopy
from pathlib import Path

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.remaining_evidence import confirm_empty_remaining
from domino_video.stone_recovery import StoneRecovery
from domino_video.validator import GameValidator
from domino_video.vision import Observation


def real_round():
    fixture = json.loads(Path("tests/fixtures/final_zero_exclusion.json").read_text("utf8"))
    rnd = fixture["round"]
    observations = []
    for item in fixture["reveal"]:
        row = Observation(item["time"], [], item["hands"], None, True, True)
        row.reveal_valid = tuple(item["valid"])
        row.reveal_points = tuple(item["points"])
        observations.append(row)
    confirm_empty_remaining([rnd], observations)
    return rnd


def test_real_final_missing_stone_uses_independent_empty_hand():
    rnd = real_round()
    assert rnd["last_active"] == 1
    events = deepcopy(rnd["events"])
    assert StoneRecovery().exclusion(rnd, events, ["0-3"])
    assert events[-1]["seat"] == 3
    assert events[-1]["stone"] == "0-3"
    proof = rnd["stone_recovery"][-1]
    assert proof["interval"] == [rnd["events"][-1]["time"], rnd["end"]]
    assert proof["time"] is None
    assert proof["region"] is None
    assert proof["evidence"]["remaining_evidence"] == rnd["remaining_evidence"]


def test_real_earlier_counter_conflict_is_not_bypassed_by_final_proof():
    rnd = real_round()
    events = deepcopy(rnd["events"])
    assert StoneRecovery().exclusion(rnd, events, ["0-3"])
    trace = []
    assert not GameReconstructor()._round_options(rnd, events, ["A", "B", "C", "D"], 2, trace)
    first = next(t for t in trace if t["states_before"] and not t["states_after"])
    assert first["stone"] == "2-4"
    assert "hand_counter_conflict" in first["rejections"]


@pytest.mark.parametrize(
    "kind",
    [
        "no_proof",
        "single",
        "duplicate_time",
        "nonzero",
        "wrong_seat",
        "two_empty",
        "unconfirmed",
        "duplicate_stone",
        "two_missing",
        "unresolved",
        "incomplete",
        "conflicting_proof",
        "before_end",
        "foreign_stone",
    ],
)
def test_final_slot_requires_complete_unambiguous_evidence(kind):
    rnd = real_round()
    proof = rnd["remaining_evidence"][0]
    missing = ["0-3"]
    if kind == "no_proof":
        rnd.pop("remaining_evidence")
    elif kind == "single":
        proof["times"] = proof["times"][:1]
        proof["points"] = [0]
    elif kind == "duplicate_time":
        proof["times"] = [proof["times"][0]] * 2
    elif kind == "nonzero":
        proof["points"][1] = 1
    elif kind == "wrong_seat":
        proof["seat"] = 2
    elif kind == "two_empty":
        rnd["remaining"][0].extend(rnd["remaining"][1])
        rnd["remaining"][1] = []
    elif kind == "unconfirmed":
        rnd["remaining_confirmed"][0] = False
    elif kind == "duplicate_stone":
        rnd["remaining"][0].append(rnd["remaining"][0][0])
    elif kind == "two_missing":
        missing.append(rnd["remaining"][0].pop())
    elif kind == "unresolved":
        rnd["unresolved"] = [dict(candidates=["6-6"])]
    elif kind == "incomplete":
        rnd["complete"] = False
    elif kind == "conflicting_proof":
        rnd["remaining_evidence"].append(dict(proof, seat=2))
    elif kind == "before_end":
        proof["times"][0] = rnd["end"] - 1
    elif kind == "foreign_stone":
        rnd["remaining"][0][0] = "7-7"
    if kind in ("single", "duplicate_time", "nonzero", "wrong_seat", "before_end"):
        # An otherwise usable old indicator must not bypass a rejected zero proof.
        rnd["last_active"] = 3
    events = deepcopy(rnd["events"])
    before = deepcopy(events)
    assert not StoneRecovery().exclusion(rnd, events, missing)
    assert events == before


def test_independent_proof_preserves_complete_validated_game(observations):
    reconstructor = GameReconstructor()
    names = next(o.names for o in observations if o.names)
    baseline = reconstructor.build(reconstructor.extract(observations), names, "withoutEggs", 50)
    rounds = reconstructor.extract(observations)
    for rnd in rounds[:2]:
        winner = rnd["last_active"]
        rnd["last_active"] = (winner + 1) % 4
        rnd["remaining_evidence"] = [
            dict(
                seat=winner + 1,
                method="zero_reveal_points",
                times=[rnd["end"] + 0.5, rnd["end"] + 1],
                points=[0, 0],
            )
        ]
    game = reconstructor.build(rounds, names, "withoutEggs", 50)
    GameValidator().validate(game)
    assert game == baseline


def test_absent_indicator_does_not_crash_or_replace_independent_proof():
    rnd = real_round()
    rnd.pop("last_active")
    assert StoneRecovery().exclusion(rnd, deepcopy(rnd["events"]), ["0-3"])
    rnd.pop("remaining_evidence")
    assert not StoneRecovery().exclusion(rnd, deepcopy(rnd["events"]), ["0-3"])
