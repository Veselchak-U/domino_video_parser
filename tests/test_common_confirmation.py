import gzip
import json
from copy import deepcopy

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def real_evidence():
    with gzip.open("tests/fixtures/common_confirmation.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    rows = []
    for row in fixture["native"]:
        for key in ("board", "uncertain_board"):
            row[key] = [StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in row[key]]
        rows.append(Observation(**row))
    return fixture["round"], rows


def test_real_shared_confirmation_recovers_right_double_before_own_move():
    rnd, native = real_evidence()
    reconstructor = GameReconstructor()
    windows = reconstructor.recovery_windows([rnd])
    assert any(start <= 80.6014667 and stop >= 86.1909 for start, stop in windows)
    reconstructor.integrate_recovery([rnd], native)
    double = next(e for e in rnd["events"] if e["stone"] == "3-3")
    own = next(e for e in rnd["events"] if e["stone"] == "3-5")
    assert double["time"] == pytest.approx(82.71823333333333)
    assert double["time"] < own["time"]
    assert double["recovery"]["evidence"].get("motion") == "incoming"
    assert 3 in double["seats"]
    assert 0 in own["seats"]
    before = deepcopy(rnd)
    reconstructor.integrate_recovery([rnd], native)
    assert rnd == before


@pytest.mark.parametrize("confirmed", [(None, None), (8.0, 8.01), (8.0, None)])
def test_absent_or_different_confirmations_do_not_create_group(confirmed):
    rnd = {
        "events": [
            dict(time=1, stone="0-0"),
            dict(time=5, stone="0-1", confirmed_at=confirmed[0]),
            dict(time=6, stone="1-1", confirmed_at=confirmed[1]),
        ]
    }
    assert StoneRecovery()._early_groups(rnd) == []


def test_shared_confirmation_alone_does_not_reorder_without_native_proof():
    rnd, _ = real_evidence()
    before = [(e["stone"], e["time"], e["seats"]) for e in rnd["events"]]
    GameReconstructor().integrate_recovery([rnd], [])
    assert [(e["stone"], e["time"], e["seats"]) for e in rnd["events"]] == before


@pytest.mark.parametrize("kind", ["single_reading", "different_neighbour"])
def test_shared_confirmation_requires_independent_placement_proof(kind):
    rnd, native = real_evidence()
    event = next(e for e in rnd["events"] if e["stone"] == "3-3")
    original_time = event["time"]
    if kind == "different_neighbour":
        event["positions"]["3-3"] = [-1000, -1000]
    else:
        kept = False
        for obs in native:
            if obs.time >= original_time:
                continue
            visible = any(tile.stone == "3-3" for tile in obs.board)
            if kept:
                obs.board = [tile for tile in obs.board if tile.stone != "3-3"]
            kept |= visible
    GameReconstructor().integrate_recovery([rnd], native)
    assert event["time"] == original_time
