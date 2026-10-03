import gzip
import json
from copy import deepcopy

import pytest
from test_stone_recovery import row

from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


@pytest.mark.parametrize(
    "stamp,stone,start,end",
    [("09-29-20-30-58", "2-2", 121, 122.2), ("10-02-16-15-52", "2-5", 115, 117)],
)
def test_real_late_board_reading_keeps_independent_hand_loss(stamp, stone, start, end):
    with gzip.open(f"tests/fixtures/late_hand_{stamp}.json.gz", "rt", encoding="utf8") as f:
        sources = json.load(f)
    rows = []
    for source in sources:
        for key in ("board", "uncertain_board"):
            source[key] = [
                StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in source[key]
            ]
        rows.append(Observation(**source))
    rnd = GameReconstructor().extract(rows)[0]
    event = next(e for e in rnd["events"] if e["stone"] == stone)
    assert start < event["time"] < end
    assert event["seats"] == [0]
    assert event["recovery"]["method"] == "hand_difference"
    assert event["recovery"]["evidence"]["late_reading"] > end


def hand_rows(transitions):
    return [row(t, hand=hand) for times, hand in transitions for t in times]


def test_late_event_uses_final_loss_after_cancel_and_is_idempotent():
    rows = hand_rows(
        [
            ([1, 1.25], ["1-6", "6-6"]),
            ([2, 2.25], ["6-6"]),
            ([3, 3.25], ["1-6", "6-6"]),
            ([4, 4.25], ["6-6"]),
        ]
    )
    rnd = dict(
        events=[dict(stone="1-6", time=8, confirmed_at=8.5, seat=1, seats=[1])], stone_recovery=[]
    )
    recovery = StoneRecovery()
    recovery._hands(rnd, rows)
    event = rnd["events"][0]
    assert event["time"] == 4
    assert event["confirmed_at"] == 8.5
    assert event["seats"] == [0]
    snapshot = deepcopy(rnd)
    recovery._hands(rnd, rows)
    assert rnd == snapshot


@pytest.mark.parametrize("times", [[2], [2, 2]])
def test_single_loss_pts_does_not_redate_existing_event(times):
    rows = hand_rows([([1, 1.25], ["1-6", "6-6"]), (times, ["6-6"])])
    rnd = dict(events=[dict(stone="1-6", time=8, seat=0, seats=[0])], stone_recovery=[])
    StoneRecovery()._hands(rnd, rows)
    assert rnd["events"][0]["time"] == 8


@pytest.mark.parametrize("event_time,returned", [(1, False), (8, True)])
def test_earlier_direct_event_and_cancelled_loss_remain_unchanged(event_time, returned):
    transitions = [([1, 1.25], ["1-6", "6-6"]), ([2, 2.25], ["6-6"])]
    if returned:
        transitions.append(([3, 3.25], ["1-6", "6-6"]))
    rnd = dict(
        events=[
            dict(stone="1-6", time=event_time, confirmed_at=event_time + 0.25, seat=0, seats=[0])
        ],
        stone_recovery=[],
    )
    StoneRecovery()._hands(rnd, hand_rows(transitions))
    assert rnd["events"][0]["time"] == event_time


def test_competing_existing_events_are_not_chosen_by_hand_loss():
    rows = hand_rows([([1, 1.25], ["1-6", "6-6"]), ([2, 2.25], ["6-6"])])
    rnd = dict(
        events=[dict(stone="1-6", time=t, seat=0, seats=[0]) for t in [8, 9]], stone_recovery=[]
    )
    StoneRecovery()._hands(rnd, rows)
    assert [e["time"] for e in rnd["events"]] == [8, 9]


def test_two_noninverse_losses_do_not_choose_first_or_last():
    rows = hand_rows(
        [
            ([1, 1.25], ["1-6", "6-6", "0-0"]),
            ([2, 2.25], ["6-6", "0-0"]),
            ([3, 3.25], ["1-6", "6-6"]),
            ([4, 4.25], ["6-6"]),
        ]
    )
    rnd = dict(events=[dict(stone="1-6", time=8, seat=0, seats=[0])], stone_recovery=[])
    StoneRecovery()._hands(rnd, rows)
    assert rnd["events"][0]["time"] == 8
