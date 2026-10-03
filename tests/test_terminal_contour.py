"""Real native evidence: final 1-1 arrives before the table darkens in 14:44."""

import gzip
import json
from copy import deepcopy

import pytest

from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def load_evidence():
    with gzip.open("tests/fixtures/terminal_contour1444.json.gz", "rt", encoding="utf8") as stream:
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
