import gzip
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from domino_video.placement_tracks import placement_tracks
from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def real_evidence():
    with gzip.open("tests/fixtures/blank_ui_competitor.json.gz", "rt", encoding="utf8") as stream:
        data = json.load(stream)
    rows = []
    for row in data["native"]:
        for field in ("board", "uncertain_board"):
            row[field] = [StoneObservation(tuple(t["values"]), tuple(t["box"])) for t in row[field]]
        rows.append(Observation(**row))
    return data["round"], rows


def test_real_blank_flight_is_not_split_by_independent_ui_blank():
    rnd, rows = real_evidence()
    recovery = StoneRecovery()
    recovery._transition_placements(rnd, rows)
    event = next(e for e in rnd["events"] if e["stone"] == "0-0")
    following = next(e for e in rnd["events"] if e["stone"] == "0-1")
    assert 303.3 < event["time"] < 303.7
    assert event["time"] < following["time"]
    assert event["recovery"]["evidence"]["motion"] == "incoming"
    before = deepcopy(rnd)
    recovery._transition_placements(rnd, rows)
    assert rnd == before


def reading(time, x, size=80):
    obs = Observation(time, [], [[], [], [], []], None, False, True)
    return obs, StoneObservation((0, 0), (x, 200, size, size // 2))


def test_independent_small_stationary_contour_does_not_split_large_flight():
    rows = [reading(t, x) for t, x in [(1, 0), (1.05, 30), (1.1, 60)]]
    rows += [reading(t, 200, 30) for t in (1, 1.05, 1.1)]
    tracks = placement_tracks(rows)
    assert sorted(len(track) for track in tracks) == [3, 3]


@pytest.mark.parametrize("kind", ["branch", "crossing"])
def test_ambiguous_physical_connections_are_not_chosen(kind):
    rows = [reading(1, 0), reading(1.05, 20), reading(1.05, 40), reading(1.1, 30)]
    if kind == "crossing":
        rows.insert(1, reading(1, 60))
    assert not any(len(track) >= 3 for track in placement_tracks(rows))


def test_large_scale_change_is_not_one_physical_track():
    tracks = placement_tracks([reading(1, 0), reading(1.05, 0, 160), reading(1.1, 0, 160)])
    assert sorted(len(track) for track in tracks) == [1, 2]


def test_repeated_pts_do_not_lengthen_physical_confirmation():
    row = reading(1, 0)
    assert [len(track) for track in placement_tracks([row, row, row])] == [1]


def test_two_real_eligible_flights_do_not_select_one():
    rnd, rows = real_evidence()
    flight = [o for o in rows if 303.39 < o.time < 303.69]
    second = [replace(o, time=o.time + 1) for o in flight]
    rnd["transition_windows"] = [[303.3, 304.8]]
    original = next(e for e in rnd["events"] if e["stone"] == "0-0")["time"]
    recovery = StoneRecovery()
    sightings = [(o, tile) for o in flight + second for tile in o.board if tile.stone == "0-0"]
    assert (
        sum(recovery._incoming(rnd, flight + second, t) for t in placement_tracks(sightings)) == 2
    )
    recovery._transition_placements(rnd, flight + second)
    assert next(e for e in rnd["events"] if e["stone"] == "0-0")["time"] == original


def test_real_blank_transition_restores_unique_round_sequence():
    rnd, rows = real_evidence()
    reconstructor = GameReconstructor()
    reconstructor.integrate_recovery([rnd], rows)
    blank = next(e for e in rnd["events"] if e["stone"] == "0-0")
    assert 303.3 < blank["time"] < 304
    assert len(reconstructor._round_options(rnd, rnd["events"], ["A", "B", "C", "D"], 3)) == 1
    previous = deepcopy(rnd)
    reconstructor.integrate_recovery([rnd], rows)
    assert rnd == previous
