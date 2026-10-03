import gzip
import json
from pathlib import Path

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def real_evidence(name):
    path = Path(__file__).parent / "fixtures" / "delayed_reordered_moves" / f"{name}.json.gz"
    with gzip.open(path, "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    rows = []
    for row in fixture["native"]:
        for key in ("board", "uncertain_board"):
            row[key] = [
                StoneObservation(tuple(tile["values"]), tuple(tile["box"])) for tile in row[key]
            ]
        for key in ("scores", "counts", "reveal_valid", "hand_regions_valid", "reveal_points"):
            if row.get(key) is not None:
                row[key] = tuple(row[key])
        rows.append(Observation(**row))
    rnd = fixture["round"]
    if fixture.get("planning_rows"):
        planning = [
            Observation(**{**row, "board": [], "uncertain_board": []})
            for row in fixture["planning_rows"]
        ]
        StoneRecovery()._turn_windows(rnd, planning)
    return rnd, rows


@pytest.mark.parametrize(
    "name,early,later,before,seat",
    [("20-31-18", "5-5", "5-6", 84.0, 0), ("13-31-48", "0-0", "2-5", 60.2, 1)],
)
def test_real_early_placement_precedes_later_chain_rotation(name, early, later, before, seat):
    """PTS/players independently checked in the source frames, not inferred from game rules."""
    rnd, native = real_evidence(name)
    GameReconstructor().integrate_recovery([rnd], native)
    first = next(event for event in rnd["events"] if event["stone"] == early)
    second = next(event for event in rnd["events"] if event["stone"] == later)
    assert first["time"] < before
    assert first["time"] < second["time"]
    assert seat in first["seats"]


def test_real_digital_transition_is_not_hidden_by_previous_late_event():
    path = Path(__file__).parent / "fixtures/delayed_reordered_moves/13-31-48.json.gz"
    with gzip.open(path, "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    rows = [
        Observation(**{**row, "board": [], "uncertain_board": []})
        for row in fixture["planning_rows"]
    ]
    rnd = fixture["round"]
    StoneRecovery()._turn_windows(rnd, rows)
    assert any(start <= 59.80427777777778 < stop for start, stop in rnd["transition_windows"])


@pytest.mark.parametrize("kind", ["known_move", "single_timer", "legacy"])
def test_transition_window_keeps_strict_suppression(kind):
    rows = [
        Observation(
            time,
            [],
            [[], [], [], []],
            seat,
            False,
            True,
            active_method=None if kind == "legacy" else "timer",
        )
        for time, seat in [(1, 1), (2, 1), (3, 1), (3.5, 2), (4, 2)]
    ]
    if kind == "single_timer":
        rows = rows[-3:]
    rnd = dict(
        start=0, end=5, complete=True, events=[dict(time=3.2 if kind == "known_move" else 1.5)]
    )
    StoneRecovery()._turn_windows(rnd, rows)
    assert not rnd["transition_windows"]
