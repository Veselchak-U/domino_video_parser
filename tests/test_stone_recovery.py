import gzip
import json
from copy import deepcopy

import pytest

from domino_video.reconstruct import GameReconstructor, ReconstructionError
from domino_video.vision import Observation, StoneObservation


def tile(stone, x=800, y=280):
    return StoneObservation(tuple(map(int, stone.split("-"))), (x, y, 50, 100))


def row(t, stones=(), hand=(), **kwargs):
    return Observation(t, list(stones), [list(hand), [], [], []], 0, False, True, **kwargs)


def test_repeated_short_tile_on_old_footprint_requests_native_reading():
    rows = [row(t, [tile("1-1")]) for t in [1, 1.5, 2]]
    rows += [row(t, [tile("1-6")]) for t in [2.5, 3]]
    rnd = GameReconstructor().extract(rows)[0]
    assert [e["candidates"] for e in rnd["unresolved"]] == [["1-6"]]


def test_native_reading_uses_continuous_settled_track_after_flight_gap():
    from domino_video.stone_recovery import StoneRecovery

    event = dict(
        time=2,
        interval=[1.5, 2],
        candidates=["1-6"],
        stone=None,
        seat=0,
        seats=[0],
        action=None,
        region=[870, 280, 50, 100],
        reason="short_observation",
    )
    rnd = dict(
        start=1,
        end=4,
        events=[dict(time=1, stone="1-1")],
        remaining=[[], [], [], []],
        unresolved=[event],
        stone_recovery=[],
    )
    rows = [row(t, [tile("1-1"), tile("1-6", 870)]) for t in [1.6, 1.62, 1.64, 2, 2.02, 2.04]]
    StoneRecovery().integrate([rnd], rows)
    assert [e["stone"] for e in rnd["events"]] == ["1-1", "1-6"]
    assert rnd["stone_recovery"][0]["evidence"]["times"] == [2, 2.02, 2.04]


def test_duplicate_pip_reading_preserves_placement_time_after_chain_moves():
    rows = [row(t, [tile("2-4", 1000), tile("2-2", 900)]) for t in [1, 1.5, 2]]
    rows += [row(t, [tile("2-4", 1000), tile("2-2", 900), tile("2-4", 830)]) for t in [3, 3.5]]
    rows += [row(t, [tile("2-4", 900), tile("2-2", 800), tile("2-6", 730)]) for t in [4, 4.5, 5]]
    rnd = GameReconstructor().extract(rows)[0]
    event = next(e for e in rnd["events"] if e["stone"] == "2-6")
    assert event["time"] == 3


def test_ambiguous_mover_uses_repeated_counters_without_global_indicator_conflict():
    reconstructor = GameReconstructor()
    deck = [f"{a}-{b}" for a in range(7) for b in range(a, 7) if (a, b) != (1, 1)]
    event = dict(time=1, stone="1-1", seats=[0, 1], seat=0, action="start")
    rnd = dict(end=5, remaining=[deck[:6], deck[6:13], deck[13:20], deck[20:]], events=[event])
    rnd["counters"] = [dict(time=event["time"] + dt, counts=(6, 0, 7, 7)) for dt in [0.4, 0.5]]
    assert reconstructor._round_options(rnd, rnd["events"], ["A", "B", "C", "D"], 1) == []


def test_recovers_hidden_move_from_stable_hand_difference():
    first = tile("1-1")
    rows = [
        row(0),
        row(1, [first], ["1-6", "6-6"]),
        row(1.25, [first], ["1-6", "6-6"]),
        row(1.5, [first], ["1-6", "6-6"]),
        row(2, [first], ["6-6"]),
        row(2.25, [first], ["6-6"]),
        row(2.5, [first], ["6-6"]),
    ]
    rnd = GameReconstructor().extract(rows)[0]
    assert [e["stone"] for e in rnd["events"]] == ["1-1", "1-6"]
    recovered = rnd["stone_recovery"][0]
    assert recovered["method"] == "hand_difference"
    assert recovered["interval"] == [1.5, 2]
    assert recovered["evidence"]["before"] == ["1-6", "6-6"]


def test_hand_reorder_and_single_frame_loss_do_not_create_move():
    first = tile("1-1")
    rows = [
        row(t, [first], hand)
        for t, hand in [
            (1, ["1-6", "6-6"]),
            (1.25, ["6-6", "1-6"]),
            (1.5, ["1-6", "6-6"]),
            (2, ["6-6"]),
            (2.25, ["1-6", "6-6"]),
            (2.5, ["1-6", "6-6"]),
        ]
    ]
    assert len(GameReconstructor().extract(rows)[0]["events"]) == 1


def test_late_reading_fills_original_slot_after_chain_moves():
    first = tile("1-1")
    unknown = StoneObservation((None, None), (870, 280, 50, 100))
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]]
    rows += [row(t, [first], uncertain_board=[unknown]) for t in [2, 2.25, 2.5]]
    rows += [row(t, [tile("1-1", 700), tile("1-6", 770)]) for t in [4, 4.25, 4.5]]
    rnd = GameReconstructor().extract(rows)[0]
    assert [e["stone"] for e in rnd["events"]] == ["1-1", "1-6"]
    assert rnd["events"][1]["time"] == 2
    assert rnd["stone_recovery"][0]["method"] == "late_reading"
    assert rnd["stone_recovery"][0]["time"] == 4


@pytest.mark.parametrize("scale,shift", [(1, 0), (0.7, -100)])
def test_occlusion_of_played_tile_does_not_create_missing_move(scale, shift):
    first, second = tile("1-1"), tile("1-6", 870)
    rows = [row(t, [first, second]) for t in [1, 1.25, 1.5]]
    unknown = StoneObservation((None, None), second.box)
    rows += [row(t, [first], uncertain_board=[unknown]) for t in [2, 2.25, 2.5]]
    moved = [
        StoneObservation(
            s.values, (int(s.box[0] * scale) + shift, 280, int(50 * scale), int(100 * scale))
        )
        for s in [first, second]
    ]
    rows += [row(t, moved) for t in [3, 3.25, 3.5]]
    rnd = GameReconstructor().extract(rows)[0]
    assert [e["stone"] for e in rnd["events"]] == ["1-1", "1-6"]
    assert rnd["unresolved"] == []
    assert rnd["stone_recovery"] == []


@pytest.mark.parametrize("reading_count,ambiguous", [(1, False), (3, True)])
def test_old_tile_occlusion_keeps_insufficient_or_ambiguous_evidence(reading_count, ambiguous):
    from domino_video.stone_recovery import StoneRecovery

    first = tile("1-1")
    second = tile("1-6", 870)
    unknown = StoneObservation((None, None), second.box)
    rows = [row(t, [first], uncertain_board=[unknown]) for t in [2, 2.25]]
    visible = [first, second]
    if ambiguous:
        visible.append(tile("1-2", 880))
    rows += [row(3 + i / 4, visible) for i in range(reading_count)]
    rnd = dict(
        start=1,
        end=4,
        events=[dict(time=1, stone=s.stone, seat=0, seats=[0], action=None) for s in visible],
    )
    StoneRecovery().augment([rnd], rows)
    assert len(rnd["unresolved"]) == 1


def test_old_layout_does_not_make_reopened_tile_ambiguous():
    from domino_video.stone_recovery import StoneRecovery

    first = tile("1-1")
    second = tile("1-6", 870)
    old = tile("1-2", 870)
    unknown = StoneObservation((None, None), second.box)
    rows = [row(t, [first, old]) for t in [1, 1.25, 1.5]]
    rows += [row(t, [first, second]) for t in [3, 3.25, 3.5]]
    rows += [row(t, [first], uncertain_board=[unknown]) for t in [4, 4.25]]
    rows += [row(t, [first, second]) for t in [5, 5.25, 5.5]]
    rnd = dict(
        start=1,
        end=6,
        events=[
            dict(time=t, stone=s.stone, seat=0, seats=[0], action=None)
            for t, s in [(1, first), (1, old), (3, second)]
        ],
    )
    StoneRecovery().augment([rnd], rows)
    assert rnd["unresolved"] == []


def test_short_last_tile_is_retained_for_additional_reading():
    first = tile("1-1")
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]] + [row(2, [first, tile("1-6", 870)])]
    rows += [Observation(2.25, [], [[], [], [], []], None, True, True)]
    rnd = GameReconstructor().extract(rows)[0]
    assert rnd["unresolved"][0]["candidates"] == ["1-6"]
    assert GameReconstructor().recovery_windows([rnd]) == [(1, 2.25)]


def test_current_recording_retains_six_double_and_short_last_tile():
    with gzip.open("tests/fixtures/occluded_observations.json.gz", "rt", encoding="utf-8") as f:
        rows = json.load(f)
    obs = [
        Observation(
            **dict(
                r, board=[StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in r["board"]]
            )
        )
        for r in rows
    ]
    rnd = GameReconstructor().extract(obs)[0]
    assert any(e["stone"] == "6-6" and 98 < e["time"] < 100 for e in rnd["events"])
    assert any("0-1" in e["candidates"] for e in rnd["unresolved"])


def test_exclusion_fills_an_internal_slot(observations):
    r = GameReconstructor()
    rounds = r.extract(observations)
    names = next(o.names for o in observations if o.names)
    r.build(rounds, names, "withoutEggs", 50)
    for rnd in rounds:
        for entry in rnd["stone_recovery"]:
            if entry["method"] == "exclusion":
                rnd["events"].append(
                    dict(
                        time=entry["interval"][1],
                        seat=entry["seat"] - 1,
                        action=None,
                        stone=entry["stone"],
                    )
                )
        rnd["events"].sort(key=lambda e: e["time"])
    expected = deepcopy(rounds[2]["events"][3])
    rounds[2]["events"][3]["stone"] = None
    rounds[2]["remaining_confirmed"] = [True] * 4
    r.build(rounds, names, "withoutEggs", 50)
    assert rounds[2]["stone_recovery"][-1]["stone"] == expected["stone"]
    assert rounds[2]["stone_recovery"][-1]["method"] == "exclusion"


@pytest.mark.parametrize("confirmed", [[False, True, True, True], [True, False, True, True]])
def test_exclusion_rejects_unconfirmed_remaining(observations, confirmed):
    r = GameReconstructor()
    rounds = r.extract(observations)
    rounds[0]["events"].pop()
    rounds[0]["remaining_confirmed"] = confirmed
    with pytest.raises(ReconstructionError, match="остат"):
        r.build(rounds, next(o.names for o in observations if o.names), "withoutEggs", 50)


def dense_rows(first, moving, times):
    return [row(t, [first, tile(moving, 870)]) for t in times]


def test_native_animation_confirms_short_final_move_once():
    r = GameReconstructor()
    first = tile("1-1")
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]] + [row(2, [first, tile("1-6", 870)])]
    rows.append(Observation(2.25, [], [[], [], [], []], None, True, True))
    rounds = r.extract(rows)
    extra = dense_rows(first, "1-6", [1.95, 2, 2.05, 2.1])
    r.integrate_recovery(rounds, extra)
    r.integrate_recovery(rounds, extra)
    assert [e["stone"] for e in rounds[0]["events"]] == ["1-1", "1-6"]
    assert len(rounds[0]["stone_recovery"]) == 1
    assert rounds[0]["stone_recovery"][0]["evidence"]["times"] == [1.95, 2, 2.05, 2.1]


def test_cancelled_animation_does_not_create_move():
    r = GameReconstructor()
    first = tile("1-1")
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]] + [row(2, [first, tile("1-6", 870)])]
    rows += [row(t, [first]) for t in [2.25, 2.5, 3]]
    rows.append(Observation(4, [], [[], [], [], []], None, True, True))
    rounds = r.extract(rows)
    extra = dense_rows(first, "1-6", [1.95, 2, 2.05, 2.1]) + [row(2.4, [first])]
    r.integrate_recovery(rounds, extra)
    assert len(rounds[0]["events"]) == 1


def test_animation_does_not_choose_between_two_slots_for_same_tile():
    r = GameReconstructor()
    first = tile("1-1")
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]] + [row(2, [first, tile("1-6", 870)])]
    rows += [Observation(2.25, [], [[], [], [], []], None, True, True)]
    rounds = r.extract(rows)
    rounds[0]["unresolved"].append(deepcopy(rounds[0]["unresolved"][0]))
    r.integrate_recovery(rounds, dense_rows(first, "1-6", [1.95, 2, 2.05, 2.1]))
    assert len(rounds[0]["events"]) == 1


def test_flight_into_occluded_slot_and_insufficient_frames():
    r = GameReconstructor()
    first = tile("1-1")
    unknown = StoneObservation((None, None), (870, 280, 50, 100))
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]]
    rows += [row(t, [first], uncertain_board=[unknown]) for t in [2, 2.25]]
    rows.append(Observation(3, [], [[], [], [], []], None, True, True))
    rounds = r.extract(rows)
    r.integrate_recovery(rounds, dense_rows(first, "1-6", [1.85, 1.9]))
    assert len(rounds[0]["events"]) == 1
    r.integrate_recovery(rounds, dense_rows(first, "1-6", [1.85, 1.9, 1.95]))
    assert rounds[0]["events"][-1]["stone"] == "1-6"


def test_multiple_hidden_slots_do_not_choose_first_match():
    r = GameReconstructor()
    first = tile("1-1")
    unknown = StoneObservation((None, None), (870, 280, 50, 100))
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]]
    rows += [row(t, [first], uncertain_board=[unknown, unknown]) for t in [2, 2.25]]
    rows += [row(t, [first, tile("1-6", 870)]) for t in [3, 3.25, 3.5]]
    rnd = r.extract(rows)[0]
    assert not any(e["method"] == "late_reading" for e in rnd["stone_recovery"])


def test_unknown_half_is_not_zero():
    assert StoneObservation((None, 6), (800, 280, 50, 100)).stone is None


def test_avatar_overlap_stays_unknown_instead_of_blank_tile():
    import cv2
    import numpy as np

    from domino_video.vision import ScreenRecognizer

    image = cv2.imread("tests/fixtures/frame_20.png")
    cream = cv2.cvtColor(np.uint8([[[25, 80, 240]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.rectangle(image, (205, 470), (254, 569), cream, -1)
    obs = ScreenRecognizer().prepare(image, 1).observation
    assert any(s.stone is None and s.box[0] == 205 for s in obs.uncertain_board)
    assert not any(s.box[0] == 205 for s in obs.board)


def test_occluded_empty_reveal_is_not_confirmed():
    first = tile("1-1")
    rows = [row(t, [first]) for t in [1, 1.25, 1.5]]
    rows += [
        Observation(
            t, [], [[], [], [], []], None, True, True, reveal_valid=(False, True, True, True)
        )
        for t in [2, 2.25]
    ]
    rnd = GameReconstructor().extract(rows)[0]
    assert rnd["remaining_confirmed"][0] is False


def load_recording_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    for value in rows:
        for field in ["board", "uncertain_board"]:
            if field in value:
                value[field] = [
                    StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in value[field]
                ]
    return [Observation(**value) for value in rows]


def test_two_fps_recording_preserves_all_deals_moves_and_results(sample_game):
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(
        load_recording_rows("tests/fixtures/two_fps_observations.json.gz")
    )
    reconstructor.integrate_recovery(
        rounds, load_recording_rows("tests/fixtures/two_fps_motion_observations.json.gz")
    )
    players = [p for team in sample_game["teams"] for p in team["players"]]
    names = [p["name"] for p in sorted(players, key=lambda p: p["seat"])]
    actual = reconstructor.build(rounds, names, "withoutEggs", 50)
    assert actual == sample_game


def test_current_recording_recovers_complete_unique_game_from_native_motion_frames():
    from domino_video.validator import GameValidator

    r = GameReconstructor()
    rounds = r.extract(load_recording_rows("tests/fixtures/occluded_observations.json.gz"))
    r.integrate_recovery(
        rounds, load_recording_rows("tests/fixtures/occluded_motion_observations.json.gz")
    )
    game = r.build(rounds, ["A", "B", "C", "D"], "withoutEggs", 50)
    GameValidator().validate(game)
    first = game["rounds"][0]
    six = [m for m in first["moves"] if m.get("stone") == "6-6"]
    assert len(six) == 1 and six[0]["player"] == "A"
    assert first["moves"][-1]["player"] == "D"
    assert set(map(int, first["moves"][-1]["stone"].split("-"))) == {0, 1}
    assert any(
        e["method"] == "animation" and e["stone"] == "0-1" for e in rounds[0]["stone_recovery"]
    )
    assert all(e["status"] == "rules_validated" for rnd in rounds for e in rnd["stone_recovery"])
    previous = [len(rnd["stone_recovery"]) for rnd in rounds]
    assert r.build(rounds, ["A", "B", "C", "D"], "withoutEggs", 50) == game
    assert [len(rnd["stone_recovery"]) for rnd in rounds] == previous


@pytest.mark.parametrize("kind", ["two_missing", "duplicate"])
def test_exclusion_does_not_repair_inconsistent_full_tile_set(observations, kind):
    r = GameReconstructor()
    rounds = r.extract(observations)
    if kind == "two_missing":
        rounds[0]["events"].pop(3)
    else:
        rounds[0]["events"].append(deepcopy(rounds[0]["events"][3]))
    with pytest.raises(ReconstructionError):
        r.build(rounds, next(o.names for o in observations if o.names), "withoutEggs", 50)


def test_new_round_does_not_reuse_previous_hand_evidence():
    first = tile("1-1")
    rows = [row(t, [first], ["1-6", "6-6"]) for t in [1, 1.25, 1.5]]
    rows += [Observation(2, [], [[], [], [], []], None, True, True), row(3)]
    rows += [row(t, [tile("2-2")], ["6-6"]) for t in [4, 4.25, 4.5]]
    rounds = GameReconstructor().extract(rows)
    assert len(rounds) == 2
    assert [e["stone"] for e in rounds[1]["events"]] == ["2-2"]


def test_exclusion_refuses_two_possible_slots(observations):
    r = GameReconstructor()
    rounds = r.extract(observations)
    rounds[0]["unresolved"] = [
        dict(
            time=t,
            interval=[t, t + 1],
            candidates=["0-3"],
            stone=None,
            seat=3,
            seats=[3],
            action=None,
        )
        for t in [70, 80]
    ]
    with pytest.raises(ReconstructionError, match="не распознаны"):
        r.build(rounds, next(o.names for o in observations if o.names), "withoutEggs", 50)


def test_candidate_budget_rejects_more_than_128_games(observations, sample_game):
    class BranchingReconstructor(GameReconstructor):
        def _round_options(self, raw, events, names, number):
            return [sample_game["rounds"][0]] * 129

    r = BranchingReconstructor()
    with pytest.raises(ReconstructionError, match="слишком много сочетаний"):
        r.build(
            r.extract(observations),
            next(o.names for o in observations if o.names),
            "withoutEggs",
            50,
        )


@pytest.mark.parametrize("angle", [25, 45, 70])
def test_reads_rotated_flying_tile(angle):
    import cv2
    import numpy as np

    from domino_video.vision import ScreenRecognizer

    image = cv2.imread("tests/fixtures/frame_20.png")
    patch = np.zeros((160, 160, 3), dtype=np.uint8)
    patch[:] = image[280, 800]
    cream = cv2.cvtColor(np.uint8([[[25, 80, 240]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    cv2.rectangle(patch, (55, 30), (104, 129), cream, -1)
    for center in [(80, 55), (67, 95), (80, 95), (92, 95), (67, 115), (80, 115), (92, 115)]:
        cv2.circle(patch, center, 4, (0, 0, 0), -1)
    patch = cv2.warpAffine(
        patch,
        cv2.getRotationMatrix2D((80, 80), angle, 1),
        (160, 160),
        borderValue=tuple(map(int, patch[0, 0])),
    )
    image[240:400, 700:860] = patch
    obs = ScreenRecognizer().prepare(image, 1, read_motion=True).observation
    assert "1-6" in [s.stone for s in obs.board]
