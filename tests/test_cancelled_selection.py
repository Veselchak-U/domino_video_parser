import gzip
import json

import pytest
from test_stone_recovery import row, tile

from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def observations():
    with gzip.open("tests/fixtures/cancelled_selection.json.gz", "rt", encoding="utf8") as stream:
        source = json.load(stream)
    result = []
    for source_row in source["coarse"]:
        for key in ("board", "uncertain_board"):
            source_row[key] = [
                StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in source_row[key]
            ]
        result.append(Observation(**source_row))
    return result


def test_real_drag_returned_to_same_hand_does_not_create_move():
    rounds = GameReconstructor().extract(observations())
    rnd = next(r for r in rounds if 266 < r["start"] < 267)
    assert "0-0" in rnd["remaining"][0]
    assert "0-0" not in [e["stone"] for e in rnd["events"]]
    actual = next(e for e in rnd["events"] if e["stone"] == "0-6")
    assert 335 < actual["time"] < 337
    assert 0 in actual["seats"]


@pytest.mark.parametrize("returned", [[3], [3, 3], [3, 3.25]])
def test_return_requires_two_distinct_observations(returned):
    rows = [row(t, hand=["1-6", "6-6"]) for t in [1, 1.25]]
    rows += [row(t, hand=["6-6"]) for t in [2, 2.25]]
    rows += [row(t, hand=["1-6", "6-6"]) for t in returned]
    rnd = dict(events=[], stone_recovery=[])
    StoneRecovery()._hands(rnd, rows)
    assert bool(rnd["events"]) == (len(set(returned)) < 2)
    assert bool(rnd["stone_recovery"]) == bool(rnd["events"])


def test_cancelled_selection_can_be_followed_by_real_play_of_same_stone():
    rows = [row(t, hand=["1-6", "6-6"]) for t in [1, 1.25]]
    rows += [row(t, hand=["6-6"]) for t in [2, 2.25]]
    rows += [row(t, hand=["1-6", "6-6"]) for t in [3, 3.25]]
    rows += [row(t, hand=["6-6"]) for t in [4, 4.25]]
    rnd = dict(events=[], stone_recovery=[])
    recovery = StoneRecovery()
    recovery._hands(rnd, rows)
    assert len(rnd["events"]) == 1
    assert rnd["events"][0]["time"] == 4
    recovery._hands(rnd, rows)
    assert len(rnd["events"]) == 1
    assert rnd["events"][0]["time"] == 4


def test_return_does_not_erase_independent_confirmed_board_event():
    first = tile("1-1")
    rows = [row(t, [first], ["1-6", "6-6"]) for t in [1, 1.25, 1.5]]
    rows += [row(t, [first, tile("1-6", 870)], ["6-6"]) for t in [2, 2.25, 2.5]]
    rows += [row(t, [first, tile("1-6", 870)], ["1-6", "6-6"]) for t in [3, 3.25]]
    rnd = GameReconstructor().extract(rows)[0]
    assert any(e["stone"] == "1-6" and e.get("confirmed_at") for e in rnd["events"])
