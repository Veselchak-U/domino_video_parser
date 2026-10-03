"""A cursor-covered moving tile must not borrow a future tile's identity."""

import gzip
import json
from copy import deepcopy

import pytest

from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def evidence(name):
    with gzip.open(f"tests/fixtures/{name}.json.gz", "rt", encoding="utf8") as stream:
        data = json.load(stream)
    rows = []
    for item in data["native"]:
        for field in ("board", "uncertain_board"):
            item[field] = [
                StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in item[field]
            ]
        rows.append(Observation(**item))
    return data["round"], rows


def test_cursor_alias_of_unplaced_own_double_does_not_create_future_move():
    rnd, rows = evidence("temporary_incoming_identity")
    future = next(e for e in rnd["events"] if e["stone"] == "3-4")
    original_time = future["time"]
    StoneRecovery().integrate([rnd], rows)
    assert future["time"] == original_time
    own = next(e for e in rnd["events"] if e["stone"] == "4-4")
    assert 136.5 < own["time"] < 137.4
    assert len([e for e in rnd["events"] if e["stone"] == "4-4"]) == 1
    before = deepcopy(rnd["events"])
    StoneRecovery().integrate([rnd], rows)
    assert rnd["events"] == before


def test_two_genuine_successive_flights_keep_distinct_values_and_times():
    # Independent actual right 3-3 followed by own 3-5 (21:23).
    # Neither is a temporary alias of the other despite the shared confirmation.
    rnd, rows = evidence("common_confirmation")
    StoneRecovery().integrate([rnd], rows)
    double = next(e for e in rnd["events"] if e["stone"] == "3-3")
    own = next(e for e in rnd["events"] if e["stone"] == "3-5")
    assert double["time"] == pytest.approx(82.71823333333333)
    assert 84 < own["time"] < 85
    assert double["time"] < own["time"]


@pytest.mark.parametrize(
    "boundary", ["wrong_hand", "single_baseline", "no_settlement", "two_contours", "broken_origin"]
)
def test_own_alias_guard_requires_all_independent_evidence(boundary):
    rnd, rows = evidence("temporary_incoming_identity")
    track = [
        (o.time, t) for o in rows if 136.56 < o.time < 136.89 for t in o.board if t.stone == "3-4"
    ]
    if boundary == "wrong_hand":
        for obs in rows:
            obs.hands[0] = ["3-4" if s == "4-4" else s for s in obs.hands[0]]
    elif boundary == "single_baseline":
        seen = False
        for obs in rows:
            if "4-4" in obs.hands[0]:
                if seen:
                    obs.hands[0].remove("4-4")
                seen = True
    elif boundary == "no_settlement":
        rows = [o for o in rows if o.time < 137.2]
    elif boundary == "two_contours":
        for obs in rows:
            if 136.898 < obs.time < 136.94:
                tiles = [t for t in obs.board if t.stone == "4-4"]
                obs.board.extend(
                    StoneObservation(t.values, (t.box[0] + 8, *t.box[1:])) for t in tiles
                )
    else:
        rows = [o for o in rows if not 136.38 < o.time < 136.55]
    assert not StoneRecovery()._own_flight_alias(rnd, rows, track)
