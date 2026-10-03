import gzip
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def load_fixture(name="transition1324"):
    with gzip.open(f"tests/fixtures/{name}.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    for key in ("coarse", "native"):
        for item in fixture[key]:
            for field in ("board", "uncertain_board"):
                item[field] = [
                    StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in item[field]
                ]
        fixture[key] = [Observation(**item) for item in fixture[key]]
    return fixture


def test_real_unread_middle_placement_gets_window_and_valid_move():
    fixture = load_fixture()
    rnd = fixture["round"]
    recovery = StoneRecovery()
    recovery.augment([rnd], fixture["coarse"])
    assert any(start < 119.7 < 120.05 < stop for start, stop in recovery.windows([rnd]))
    recovery.integrate([rnd], fixture["native"])
    event = next(e for e in rnd["events"] if e["stone"] == "3-3")
    assert 119.7 < event["time"] < 120
    assert event["seats"] == [1]
    assert [e["stone"] for e in rnd["events"]][-5:] == ["4-6", "3-3", "2-3", "3-4", "2-6"]
    options = GameReconstructor()._round_options(rnd, rnd["events"], ["A", "B", "C", "D"], 1)
    assert len(options) == 1
    moves = [m for m in options[0]["moves"] if m.get("stone") in ("3-3", "2-6", "6-2")]
    assert moves[0]["player"] == "B"
    assert moves[1]["player"] == "D"
    original = deepcopy(rnd["events"])
    recovery.integrate([rnd], fixture["native"])
    assert rnd["events"] == original


def test_real_pass_transition_does_not_create_move():
    fixture = load_fixture()
    rnd = fixture["round"]
    recovery = StoneRecovery()
    recovery.augment([rnd], fixture["coarse"])
    original = deepcopy(rnd["events"])
    recovery.integrate([rnd], [o for o in fixture["native"] if o.time > 120.67])
    assert [(e["time"], e["stone"]) for e in rnd["events"]] == [
        (e["time"], e["stone"]) for e in original
    ]


def test_duplicate_timestamps_do_not_establish_transition():
    fixture = load_fixture()
    rnd = fixture["round"]
    before = next(o for o in fixture["coarse"] if 119 < o.time < 119.2)
    after = next(o for o in fixture["coarse"] if 120.1 < o.time < 120.3)
    StoneRecovery().augment([rnd], [before, before, after, after])
    assert not rnd["transition_windows"]


def test_existing_incoming_proof_is_not_silently_replaced():
    fixture = load_fixture()
    rnd = fixture["round"]
    rnd["events"].append(
        dict(
            stone="3-3",
            time=130.0,
            seats=[1],
            seat=1,
            recovery=dict(method="animation", evidence=dict(motion="incoming")),
        )
    )
    recovery = StoneRecovery()
    recovery.augment([rnd], fixture["coarse"])
    recovery.integrate([rnd], fixture["native"])
    assert next(e for e in rnd["events"] if e["stone"] == "3-3")["time"] == 130.0


def test_stationary_track_cannot_create_middle_move():
    fixture = load_fixture()
    rnd = fixture["round"]
    recovery = StoneRecovery()
    recovery.augment([rnd], fixture["coarse"])
    recovery.integrate([rnd], [o for o in fixture["native"] if o.time >= 120.05])
    assert "3-3" not in {e["stone"] for e in rnd["events"]}


def test_unreliable_indicator_does_not_assign_middle_player():
    fixture = load_fixture()
    rnd = fixture["round"]
    recovery = StoneRecovery()
    recovery.augment([rnd], fixture["coarse"])
    rnd["indicator_unreliable"] = True
    recovery.integrate([rnd], fixture["native"])
    assert next(e for e in rnd["events"] if e["stone"] == "3-3")["seats"] == [0, 1, 2, 3]


@pytest.mark.parametrize("boundary", ["remaining", "no_anchor", "two_flights"])
def test_unsupported_middle_identity_is_rejected(boundary):
    fixture = load_fixture()
    rnd, rows = fixture["round"], fixture["native"]
    recovery = StoneRecovery()
    recovery.augment([rnd], fixture["coarse"])
    if boundary == "remaining":
        rnd["remaining"][0].append("3-3")
    elif boundary == "no_anchor":
        rows = [replace(o, board=[s for s in o.board if s.stone == "3-3"]) for o in rows]
    else:
        rows = [o for o in rows if o.time < 120.2]
        rows += [replace(o, time=o.time + 0.8) for o in rows if 119.7 < o.time < 120.2]
        rnd["transition_windows"] = [[119, 121]]
    recovery.integrate([rnd], rows)
    assert "3-3" not in {e["stone"] for e in rnd["events"]}


def test_accounted_late_stationary_animation_can_be_redated():
    fixture = load_fixture()
    rnd = fixture["round"]
    rnd["events"].append(
        dict(
            stone="3-3",
            time=130.0,
            seats=[1],
            seat=1,
            recovery=dict(method="animation", evidence=dict(times=[130, 130.05, 130.1])),
        )
    )
    recovery = StoneRecovery()
    recovery.augment([rnd], fixture["coarse"])
    assert (
        len({e["stone"] for e in rnd["events"]} | {s for hand in rnd["remaining"] for s in hand})
        == 28
    )
    recovery.integrate([rnd], fixture["native"])
    event = next(e for e in rnd["events"] if e["stone"] == "3-3")
    assert 119.7 < event["time"] < 120
    assert event["recovery"]["evidence"]["late_reading"] == 130.0


@pytest.mark.parametrize("boundary", ["single", "flicker", "long_gap", "incomplete", "known_move"])
def test_transition_window_requires_unexplained_stable_change(boundary):
    fixture = load_fixture()
    rnd = fixture["round"]
    rows = [o for o in fixture["coarse"] if 118.6 < o.time < 121.3]
    if boundary == "single":
        rows = [rows[0], rows[-1]]
    elif boundary == "flicker":
        rows = [replace(o, active=i % 2) for i, o in enumerate(rows)]
    elif boundary == "long_gap":
        rows = [replace(o, time=o.time + 5) if o.active == 2 else o for o in rows]
    elif boundary == "incomplete":
        rnd["complete"] = False
    else:
        rnd["events"].append(dict(stone="3-3", time=119.9, seats=[1], seat=1))
    StoneRecovery().augment([rnd], rows)
    assert not rnd.get("transition_windows")
