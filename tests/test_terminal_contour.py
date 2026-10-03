"""Real native evidence: final 1-1 arrives before the table darkens in 14:44."""

import gzip
import json
from copy import deepcopy

import pytest

from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def load_evidence(name="terminal_contour1444"):
    with gzip.open(f"tests/fixtures/{name}.json.gz", "rt", encoding="utf8") as stream:
        data = json.load(stream)
    for key in ("native", "fragment"):
        for item in data[key]:
            for field in ("board", "uncertain_board"):
                item[field] = [
                    StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in item[field]
                ]
        data[key] = [Observation(**item) for item in data[key]]
    return data


def test_real_final_flight_continues_into_partly_hidden_settled_contour():
    data = load_evidence()
    rnd = data["round"]
    recovery = StoneRecovery()
    recovery.integrate([rnd], data["fragment"] + data["native"])
    moves = [e for e in rnd["events"] if e["stone"] == "1-1"]
    assert len(moves) == 1
    event = moves[0]
    assert event["time"] == pytest.approx(183.85204444444443)
    assert event["time"] > next(e["time"] for e in rnd["events"] if e["stone"] == "3-4")
    assert 3 in event["seats"]
    original = deepcopy(rnd["events"])
    recovery.integrate([rnd], data["fragment"] + data["native"])
    assert rnd["events"] == original


def test_real_repeated_fragment_of_old_tile_cannot_be_exclusion_slot():
    data = load_evidence()
    rnd = data["round"]
    StoneRecovery().integrate([rnd], data["fragment"])
    assert not any(e.get("region") == [1341, 179, 29, 53] for e in rnd["unresolved"])
    assert not any(e["stone"] == "1-1" for e in rnd["events"])


@pytest.mark.parametrize("boundary", ["one_settled", "competing_contour", "no_neighbor"])
def test_final_contour_requires_unambiguous_repeated_installation(boundary):
    data = load_evidence()
    rnd, rows = data["round"], data["native"]
    if boundary == "one_settled":
        rows = [o for o in rows if o.time < 184.16]
    elif boundary == "competing_contour":
        for obs in rows:
            if 184.15 < obs.time < 184.19:
                tile = next(s for s in obs.board if s.stone == "0-1" and s.box[0] < 400)
                x, y, w, h = tile.box
                obs.uncertain_board.append(StoneObservation((None, None), (x + 8, y, w, h)))
    else:
        for obs in rows:
            obs.board = [s for s in obs.board if s.stone != "1-6"]
    StoneRecovery().integrate([rnd], rows)
    assert not any(e["stone"] == "1-1" for e in rnd["events"])


def test_real_stable_terminal_segment_survives_later_singleton_and_remote_alias():
    data = load_evidence("terminal_contour2030")
    rnd = data["round"]
    StoneRecovery().integrate([rnd], data["native"])
    event = next(e for e in rnd["events"] if e["stone"] == "1-2")
    assert event["time"] < 143.92
    assert event["time"] != pytest.approx(144.09743333333333)
    assert 0 in event["seats"]


@pytest.mark.parametrize(
    "boundary",
    ["alias_without_history", "one_flight_reading", "unplayed_conflict", "duplicate_anchors"],
)
def test_hidden_final_value_needs_independent_identity_proof(boundary):
    data = load_evidence()
    rnd, rows = data["round"], data["native"]
    for obs in rows:
        if boundary == "alias_without_history" and obs.time < 183.852:
            obs.board = [t for t in obs.board if t.stone != "0-1"]
        elif boundary == "one_flight_reading" and obs.time < 184.11:
            obs.board = [t for t in obs.board if t.stone != "1-1"]
        elif boundary == "duplicate_anchors" and obs.time < 183.852:
            obs.board += [
                StoneObservation(t.values, (t.box[0] + 30, *t.box[1:]))
                for t in list(obs.board)
                if t.stone in {"0-3", "3-4"}
            ]
        elif boundary == "unplayed_conflict" and 184.15 < obs.time < 184.19:
            obs.board = [
                StoneObservation((0, 0), t.box) if t.stone == "0-1" and t.box[0] < 400 else t
                for t in obs.board
            ]
    StoneRecovery().integrate([rnd], rows)
    assert not any(e["stone"] == "1-1" for e in rnd["events"])


@pytest.mark.parametrize(
    "boundary", ["one_prior", "moving_anchors", "moving_fragment", "full_footprint"]
)
def test_old_fragment_requires_repeated_stationary_containment(boundary):
    data = load_evidence()
    rnd, rows = data["round"], data["fragment"]
    slot = rnd["unresolved"][0]
    if boundary == "one_prior":
        before = [o for o in rows if o.time < slot["time"]]
        rows = before[:1] + [o for o in rows if o.time >= slot["time"]]
    elif boundary == "full_footprint":
        slot["region"] = [1261, 179, 109, 53]
    else:
        for obs in rows:
            if obs.time < slot["time"]:
                continue
            if boundary == "moving_anchors":
                obs.board = [
                    StoneObservation(t.values, (t.box[0] + 12, *t.box[1:])) for t in obs.board
                ]
            else:
                obs.uncertain_board = [
                    StoneObservation(t.values, (t.box[0] + 12, *t.box[1:]))
                    for t in obs.uncertain_board
                ]
    StoneRecovery().integrate([rnd], rows)
    assert any(e["time"] == slot["time"] for e in rnd["unresolved"])


def test_remote_false_value_does_not_replace_a_missing_terminal_segment():
    data = load_evidence("terminal_contour2030")
    rnd, rows = data["round"], data["native"]
    for obs in rows:
        obs.board = [t for t in obs.board if not (t.stone == "1-2" and t.box[0] < 1200)]
    StoneRecovery().integrate([rnd], rows)
    assert not any(e["stone"] == "1-2" for e in rnd["events"])
