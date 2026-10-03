import gzip
import json
from copy import deepcopy

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation, StoneObservation


def rows(values):
    result = []
    for value in deepcopy(values):
        for field in ("board", "uncertain_board"):
            value[field] = [
                StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in value[field]
            ]
        result.append(Observation(**value))
    return result


@pytest.fixture
def recording():
    with gzip.open("tests/fixtures/late_chronology.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    return rows(fixture["ordinary"]), rows(fixture["native"])


def test_layout_change_requests_hidden_singleton_interval(recording):
    ordinary, _ = recording
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(ordinary)
    assert any(
        start <= 365.78 and end >= 371.44
        for start, end in reconstructor.recovery_windows([rounds[3]])
    )


@pytest.mark.parametrize(
    "number,stone,time,seat",
    [
        (1, "3-3", 158.9877888888889, 3),
        (3, "3-6", 365.78073333333333, 1),
        (3, "0-6", 370.6034222222222, 3),
    ],
)
def test_real_hidden_placement_keeps_visual_time_and_player(recording, number, stone, time, seat):
    ordinary, native = recording
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(ordinary)
    reconstructor.integrate_recovery(rounds, native)
    events = [e for e in rounds[number]["events"] if e["stone"] == stone]
    assert len(events) == 1
    assert events[0]["time"] == pytest.approx(time)
    assert seat in events[0]["seats"]
    assert 0 not in events[0]["seats"]
    assert events[0]["recovery"]["evidence"]["times"][0] == pytest.approx(time)
    before = deepcopy(rounds[number])
    reconstructor.integrate_recovery(rounds, native)
    assert rounds[number] == before


def test_terminal_window_preserves_indicator_before_penultimate_flight():
    with gzip.open("tests/fixtures/late_chronology.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(rows(fixture["ordinary"]))
    windows = reconstructor.recovery_windows([rounds[0]])
    native = [
        o
        for o in rows(fixture["terminal_native"])
        if any(start <= o.time <= stop for start, stop in windows)
    ]
    reconstructor.integrate_recovery(rounds, native)
    event = next(e for e in rounds[0]["events"] if e["stone"] == "2-5")
    # The left player's flight is independently visible at 82.906 s; the
    # indicator has already advanced to the upper player at the late reading.
    assert 1 in event["seats"]
    assert event["time"] < next(e["time"] for e in rounds[0]["events"] if e["stone"] == "0-4")


@pytest.mark.parametrize("evidence", ["complete", "single", "wrong_neighbor"])
def test_real_delayed_cluster_uses_existing_window_for_later_reading(evidence):
    with gzip.open("tests/fixtures/late_chronology.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    rounds = fixture["cluster_rounds"]
    native = rows(fixture["cluster_native"])
    event = next(e for e in rounds[3]["events"] if e["stone"] == "0-1")
    original_time = event["time"]
    if evidence == "single":
        found = False
        for obs in native:
            if obs.time < original_time:
                matches = [s for s in obs.board if s.stone == "0-1"]
                if found:
                    obs.board = [s for s in obs.board if s.stone != "0-1"]
                found |= bool(matches)
    elif evidence == "wrong_neighbor":
        event["positions"]["0-1"] = [-1000, -1000]
    reconstructor = GameReconstructor()
    reconstructor.integrate_recovery(rounds, native)
    if evidence != "complete":
        assert event["time"] == original_time
        return
    assert event["time"] == pytest.approx(386.5944111111111)
    assert 1 in event["seats"]
    assert event["time"] < next(e["time"] for e in rounds[3]["events"] if e["stone"] == "0-0")
    assert next(e["time"] for e in rounds[2]["events"] if e["stone"] == "0-5") == pytest.approx(
        278.54116666666664
    )
    before = deepcopy(rounds)
    reconstructor.integrate_recovery(rounds, native)
    assert rounds == before
    game = reconstructor.build(rounds, list("ABCD"), "withoutEggs", 50)
    assert len(game["rounds"]) == 5
    assert game["result"]["winner"] == "Team B"
