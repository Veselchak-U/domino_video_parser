import gzip
import json
from copy import deepcopy

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.validator import GameValidator
from domino_video.vision import Observation, StoneObservation


def test_flag_occlusion_does_not_turn_two_six_into_one_six():
    import cv2
    import numpy as np

    from domino_video.vision import ScreenRecognizer

    image = np.zeros((720, 1608, 3), dtype=np.uint8)
    image[:] = (60, 120, 30)
    image[380:525, 210:305] = cv2.imread("tests/fixtures/flag_occluded_tile.png")
    obs = ScreenRecognizer().prepare(image, 286.8765333333333, False).observation
    assert not obs.board
    assert obs.uncertain_board


def test_flag_touching_border_keeps_readable_two_six():
    import cv2
    import numpy as np

    from domino_video.vision import ScreenRecognizer

    image = np.zeros((720, 1608, 3), dtype=np.uint8)
    image[:] = (60, 120, 30)
    image[370:525, 210:320] = cv2.imread("tests/fixtures/flag_border_tile.png")
    obs = ScreenRecognizer().prepare(image, 47.49338888888889, False).observation
    assert [s.stone for s in obs.board] == ["2-6"]


def test_flag_beside_center_pip_keeps_readable_zero_one():
    import cv2
    import numpy as np

    from domino_video.vision import ScreenRecognizer

    image = np.zeros((720, 1608, 3), dtype=np.uint8)
    image[:] = (60, 120, 30)
    image[410:525, 235:300] = cv2.imread("tests/fixtures/flag_readable_tile.png")
    obs = ScreenRecognizer().prepare(image, 168.83818888888888, False).observation
    assert [s.stone for s in obs.board] == ["0-1"]


def tile(stone, x):
    return StoneObservation(tuple(map(int, stone.split("-"))), (x, 280, 50, 100))


def observation(time, x):
    return Observation(time, [tile("1-1", 800), tile("1-6", x)], [[], [], [], []], 0, False, True)


@pytest.mark.parametrize("early, x, expected", [(2, 870, 3), (1, 870, 8), (2, 1300, 8)])
def test_early_placement_time_survives_occlusion_only_with_repeated_chain_evidence(
    early, x, expected
):
    rows = [
        Observation(t, [tile("1-1", 800)], [[], [], [], []], 0, False, True) for t in [1, 1.5, 2]
    ]
    rows += [observation(3 + i * 0.5, x) for i in range(early)]
    rows += [
        Observation(t, [tile("1-1", 800)], [[], [], [], []], 0, False, True) for t in [4, 5, 6, 7]
    ]
    rows += [observation(t, 870) for t in [8, 8.5, 9]]
    rnd = GameReconstructor().extract(rows)[0]
    event = next(e for e in rnd["events"] if e["stone"] == "1-6")
    assert event["time"] == expected


@pytest.mark.parametrize("old", ["earlier", "elsewhere", "matching", "ambiguous", "none"])
def test_terminal_placement_matches_only_compatible_slots(old):
    event = dict(
        time=4.8,
        interval=[4.7, 4.8],
        stone=None,
        candidates=[],
        seat=0,
        seats=[0],
        action=None,
        region=[870, 280, 50, 100],
        reason="occluded",
    )
    if old == "earlier":
        event.update(time=2, interval=[1.5, 2])
    if old == "elsewhere":
        event["region"] = [400, 280, 50, 100]
    slots = [] if old == "none" else [event]
    if old == "ambiguous":
        slots.append(deepcopy(event))
    rnd = dict(
        start=1,
        end=5,
        complete=True,
        events=[dict(time=1, stone="1-1")],
        remaining=[[], [], [], []],
        unresolved=slots,
        stone_recovery=[],
        last_active=0,
    )
    rows = [
        observation(t, x)
        for t, x in [(4.6, 1150), (4.7, 1000), (4.8, 870), (4.82, 870), (4.84, 870)]
    ]
    r = GameReconstructor()
    r.integrate_recovery([rnd], rows)
    r.integrate_recovery([rnd], rows)
    assert [e["stone"] for e in rnd["events"]] == (
        ["1-1"] if old == "ambiguous" else ["1-1", "1-6"]
    )
    assert (
        len(rnd["unresolved"])
        == {"earlier": 1, "elsewhere": 1, "matching": 0, "ambiguous": 2, "none": 0}[old]
    )
    if old != "ambiguous":
        assert rnd["events"][-1]["positions"]["1-6"] == [895, 330]


@pytest.mark.parametrize("slot_count", [1, 2])
@pytest.mark.parametrize("chain_visible", [True, False])
def test_native_track_dates_late_reading_without_duplicating_move(slot_count, chain_visible):
    slot = dict(
        time=2,
        interval=[1.5, 2],
        stone=None,
        candidates=[],
        seat=0,
        seats=[0],
        action=None,
        region=[870, 280, 50, 100],
        reason="occluded",
    )
    rnd = dict(
        start=1,
        end=6,
        complete=False,
        events=[dict(time=1, stone="1-1"), dict(time=4, stone="1-6", seat=0, seats=[0])],
        unresolved=[deepcopy(slot) for _ in range(slot_count)],
        remaining=None,
        stone_recovery=[],
    )
    rows = [observation(t, 870) for t in [1.9, 1.92, 1.94]]
    if not chain_visible:
        for obs in rows:
            obs.board = [s for s in obs.board if s.stone == "1-6"]
    r = GameReconstructor()
    r.integrate_recovery([rnd], rows)
    r.integrate_recovery([rnd], rows)
    assert [e["stone"] for e in rnd["events"]] == ["1-1", "1-6"]
    resolved = slot_count == 1 and chain_visible
    assert rnd["events"][-1]["time"] == (1.9 if resolved else 4)
    assert len(rnd["unresolved"]) == (0 if resolved else slot_count)


def test_real_terminal_zero_three_with_stale_occlusions():
    # Real 20:45 recording, source SHA-256:
    # 1893ec38255aefb16c9c80780be2d7948c3174f6d8598fc416511196ab270dfe.
    with gzip.open("tests/fixtures/terminal_placement.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)

    def observations(values):
        rows = []
        for value in values:
            for field in ["board", "uncertain_board"]:
                value[field] = [
                    StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in value[field]
                ]
            rows.append(Observation(**value))
        return rows

    rows = observations(fixture["motion"])
    r = GameReconstructor()
    rounds = r.extract(observations(fixture["observations"]))
    r.integrate_recovery(rounds, rows)
    game = r.build(rounds, ["A", "B", "C", "D"], "withoutEggs", 50)
    GameValidator().validate(game)
    assert game["rounds"][0]["moves"][-1] == dict(player="D", action="left", stone="3-0")
    proof = next(e for e in rounds[0]["stone_recovery"] if e["stone"] == "0-3")
    assert proof["method"] == "animation"
    assert proof["status"] == "rules_validated"
    assert len(proof["evidence"]["times"]) >= 3
    assert 113.17 < proof["time"] < 113.93
    r.integrate_recovery(rounds, rows)
    assert r.build(rounds, ["A", "B", "C", "D"], "withoutEggs", 50) == game
