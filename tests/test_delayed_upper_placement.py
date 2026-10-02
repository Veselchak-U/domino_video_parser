import gzip
import json
from copy import deepcopy

import cv2
import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.validator import GameValidator
from domino_video.vision import Observation, ScreenRecognizer, StoneObservation


def scenario(kind="incoming"):
    neighbor = StoneObservation((3, 5), (350, 150, 100, 50))
    event = dict(
        time=10,
        stone="4-5",
        seat=3,
        seats=[3],
        action="right",
        positions={"4-5": [300, 175], "3-5": [400, 175]},
        tile_size=100,
    )
    rnd = dict(
        start=2,
        end=12,
        complete=True,
        events=[
            dict(time=2, stone="3-5", seat=1, seats=[1], action="start"),
            event,
            dict(time=10, stone="2-6", seat=3, seats=[3], action="left"),
        ],
        remaining=None,
        unresolved=[],
        stone_recovery=[],
    )
    rows = [Observation(t, [neighbor], [[], [], [], []], 2, False, True) for t in [3, 3.4]]
    times = [4, 4.05, 4.1]
    xs = [100, 150, 200]
    if kind == "away":
        xs.reverse()
    elif kind == "broken":
        times = [4, 4.5, 5]
    elif kind == "single":
        times, xs = times[:1], xs[:1]
    elif kind == "duplicate_time":
        times = [4, 4, 4]
    elif kind == "returned":
        times = [4, 4.05, 4.1, 4.15, 4.2]
        xs = [100, 150, 200, 150, 100]
    elif kind == "wrong_neighbor":
        event["positions"]["3-5"] = [1200, 175]
    elif kind == "settled":
        xs = [250, 250, 250]
    for t, x in zip(times, xs):
        rows.append(
            Observation(
                t,
                [neighbor, StoneObservation((4, 5), (x, 150, 100, 50))],
                [[], [], [], []],
                None,
                False,
                True,
            )
        )
    if kind == "two_tracks":
        for row in deepcopy(rows[2:]):
            row.time += 2
            rows.append(row)
    return rnd, rows


def test_coincident_readings_request_the_preceding_interval():
    rnd, _ = scenario()
    reconstructor = GameReconstructor()
    assert any(start <= 3 and end >= 10 for start, end in reconstructor.recovery_windows([rnd]))
    rnd["events"][-1]["time"] = 11
    assert reconstructor.recovery_windows([rnd]) == []


@pytest.mark.parametrize("kind", ["incoming", "settled"])
def test_confirmed_early_track_dates_existing_event_once(kind):
    rnd, rows = scenario(kind)
    reconstructor = GameReconstructor()
    reconstructor.integrate_recovery([rnd], rows)
    event = rnd["events"][1]
    assert (event["stone"], event["time"], event["seats"]) == ("4-5", 4, [2])
    assert len(rnd["events"]) == 3
    assert event["recovery"]["evidence"]["times"] == [4, 4.05, 4.1]
    before = deepcopy(rnd)
    reconstructor.integrate_recovery([rnd], rows)
    assert rnd == before


@pytest.mark.parametrize(
    "kind",
    ["away", "returned", "broken", "single", "duplicate_time", "wrong_neighbor", "two_tracks"],
)
def test_unconfirmed_track_does_not_redate_later_reading(kind):
    rnd, rows = scenario(kind)
    GameReconstructor().integrate_recovery([rnd], rows)
    event = next(e for e in rnd["events"] if e["stone"] == "4-5")
    assert (event["time"], event["seats"]) == (10, [3])
    assert rnd["stone_recovery"] == []


@pytest.mark.parametrize("unreliable,visible", [(True, True), (True, False), (False, False)])
def test_early_track_preserves_uncertainty_of_the_player_indicator(unreliable, visible):
    rnd, rows = scenario()
    rnd["indicator_unreliable"] = unreliable
    if not visible:
        for row in rows:
            row.active = None
    GameReconstructor().integrate_recovery([rnd], rows)
    event = next(e for e in rnd["events"] if e["stone"] == "4-5")
    if unreliable:
        assert event["seats"] == [0, 1, 2, 3]
        assert event["time"] == 4
    else:
        assert event["seats"] == [3]
        assert event["time"] == 10


def test_native_upper_flight_is_available_only_to_targeted_reading():
    image = cv2.imread("tests/fixtures/upper_flight.png")
    assert image is not None
    recognizer = ScreenRecognizer()
    try:
        ordinary = recognizer.prepare(image, 60.94425555555556, read_text=False).observation
        motion = recognizer.prepare(
            image, 60.94425555555556, read_text=False, read_motion=True
        ).observation
    finally:
        recognizer.close()
    assert "4-5" not in [s.stone for s in ordinary.board]
    assert "4-5" in [s.stone for s in motion.board]


@pytest.mark.parametrize("continuous", [True, False])
def test_terminal_reading_links_existing_short_event_only_through_continuous_track(continuous):
    old = dict(
        time=2.2,
        interval=[2, 2.2],
        stone=None,
        candidates=["1-6"],
        seat=1,
        seats=[1],
        action=None,
        reason="short_observation",
        region=[870, 280, 50, 100],
    )
    rnd = dict(
        start=1,
        end=4,
        complete=True,
        remaining=[[], [], [], []],
        events=[dict(time=1, stone="1-1", seat=0, seats=[0], action="start")],
        unresolved=[old],
        stone_recovery=[],
    )
    times = [round(2.1 + i * 0.05, 2) for i in range(37)]
    if not continuous:
        times = [t for t in times if t < 2.6 or t > 3]
    rows = [
        Observation(
            t,
            [
                StoneObservation((1, 1), (800, 280, 50, 100)),
                StoneObservation((1, 6), (870, 280, 50, 100)),
            ],
            [[], [], [], []],
            1,
            False,
            True,
        )
        for t in times
    ]
    GameReconstructor().integrate_recovery([rnd], rows)
    if continuous:
        assert [e["stone"] for e in rnd["events"]] == ["1-1", "1-6"]
        assert rnd["events"][-1]["time"] == 2.1
        assert rnd["unresolved"] == []
    else:
        assert old in rnd["unresolved"]
        assert not any(e["stone"] == "1-6" and e["time"] == 2.1 for e in rnd["events"])


def test_real_upper_placement_and_penultimate_double_are_recovered_once():
    with gzip.open(
        "tests/fixtures/delayed_upper_placement.json.gz", "rt", encoding="utf8"
    ) as stream:
        fixture = json.load(stream)

    def observations(values):
        result = []
        for value in values:
            for field in ("board", "uncertain_board"):
                value[field] = [
                    StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in value[field]
                ]
            result.append(Observation(**value))
        return result

    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(observations(fixture["observations"]))
    motion = observations(fixture["motion"])
    reconstructor.integrate_recovery(rounds, motion)
    game = reconstructor.build(rounds, list("ABCD"), "withoutEggs", 50)
    GameValidator().validate(game)
    assert [len(r["moves"]) for r in game["rounds"]] == [23, 27]
    assert dict(player="C", action="right", stone="5-4") in game["rounds"][0]["moves"]
    assert dict(player="D", action="left", stone="6-2") in game["rounds"][0]["moves"]
    assert game["rounds"][1]["moves"][-2:] == [
        dict(player="B", action="right", stone="1-1"),
        dict(player="C", action="right", stone="1-6"),
    ]
    assert game["rounds"][0]["result"]["reason"] == "fish"
    assert game["rounds"][1]["result"]["total_score"] == {"Team A": 14, "Team B": 76}
    proof = next(p for p in rounds[0]["stone_recovery"] if p["stone"] == "4-5")
    assert proof["status"] == "rules_validated"
    assert 60.9 < proof["time"] < 61
    assert len(proof["evidence"]["times"]) == 6
    reconstructor.integrate_recovery(rounds, motion)
    assert reconstructor.build(rounds, list("ABCD"), "withoutEggs", 50) == game
