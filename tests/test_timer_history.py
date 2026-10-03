import pytest

from domino_video.indicator_evidence import mark_weak_indicator, preserve_player_candidates
from domino_video.recognition_plan import recognition_requests
from domino_video.vision import Observation


def timer(time, seat, value):
    obs = Observation(time, [], [[], [], [], []], seat, False, True, active_method="timer")
    obs.timer_value = value
    return obs


def test_timer_history_localizes_weak_color_to_other_events():
    first = dict(time=10, stone="1-2", seat=None, seats=[0, 1, 2, 3])
    first["recovery"] = dict(method="animation", evidence=dict(motion="incoming"))
    second = dict(time=15, stone="2-3", seat=None, seats=[0, 1, 2, 3])
    rnd = dict(start=10, end=20, complete=True, events=[first, second], indicator_unreliable=True)
    rows = [timer(9, 2, 23), timer(9.5, 2, 22)]
    mark_weak_indicator(rnd, rows)
    preserve_player_candidates(rnd)
    assert first["seats"] == [2]
    assert first["timer_confirmed"]
    assert second["seats"] == [0, 1, 2, 3]
    # Only the uncertain second event needs counter OCR.
    rows.extend(Observation(t, [], [[], [], [], []], None, False, True) for t in (11, 12, 16, 17))
    request = recognition_requests([rnd], rows)
    assert not any(request.get(t) and request[t].counts for t in (11, 12))
    assert all(request[t].counts for t in (16, 17))


def test_conflicting_or_stale_timer_keeps_candidates_unknown():
    for rows in ([timer(9, 1, 23), timer(9.5, 2, 23)], [timer(5, 2, 23), timer(5.5, 2, 22)]):
        event = dict(time=10, stone="1-2", seat=None, seats=[0, 1, 2, 3])
        rnd = dict(indicator_unreliable=True, events=[event])
        mark_weak_indicator(rnd, rows)
        preserve_player_candidates(rnd)
        assert event["seats"] == [0, 1, 2, 3]


def test_repeated_integration_does_not_duplicate_timer_evidence():
    rnd = dict(events=[])
    rows = [timer(1, 1, 23), timer(1.5, 1, 22)]
    mark_weak_indicator(rnd, rows)
    mark_weak_indicator(rnd, rows)
    assert len(rnd["timer_evidence"]) == 2


def test_later_ambiguous_timer_blocks_earlier_agreement():
    rnd = dict(
        indicator_unreliable=True,
        events=[dict(time=11, stone="1-2", seat=None, seats=[0, 1, 2, 3])],
    )
    rows = [timer(9.7, 0, 20), timer(10.1, 0, 20)]
    rows.append(
        Observation(10.4, [], [[], [], [], []], None, False, True, active_method="ambiguous_timer")
    )
    mark_weak_indicator(rnd, rows)
    preserve_player_candidates(rnd)
    assert not rnd["events"][0].get("timer_confirmed")
    assert rnd["events"][0]["seats"] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "readings",
    [
        [(5, 2, 23), (5.5, 2, 22)],
        [(9, 1, 23), (9.5, 2, 23)],
    ],
    ids=["stale", "conflicting"],
)
def test_unconfirmed_digital_history_clears_actor_without_weak_color(readings):
    event = dict(time=10, stone="1-2", seat=2, seats=[2])
    rnd = dict(events=[event])
    mark_weak_indicator(rnd, [timer(*reading) for reading in readings])
    preserve_player_candidates(rnd)
    assert event["seat"] is None
    assert event["seats"] == [0, 1, 2, 3]
    assert not event.get("timer_confirmed")


def test_legacy_observations_without_timer_keep_existing_actor():
    event = dict(time=10, stone="1-2", seat=2, seats=[2])
    rnd = dict(events=[event])
    legacy = Observation(9.5, [], [[], [], [], []], 2, False, True)
    mark_weak_indicator(rnd, [legacy])
    preserve_player_candidates(rnd)
    assert (event["seat"], event["seats"]) == (2, [2])


def test_independent_hand_difference_survives_stale_digital_history():
    proof = dict(method="hand_difference", stone="1-2", seat=1)
    event = dict(time=10, stone="1-2", seat=2, seats=[2])
    rnd = dict(events=[event], stone_recovery=[proof])
    mark_weak_indicator(rnd, [timer(5, 2, 23), timer(5.5, 2, 22)])
    preserve_player_candidates(rnd)
    assert (event["seat"], event["seats"]) == (0, [0])
    assert not event.get("timer_confirmed")


@pytest.mark.parametrize("switch_time,confirmed", [(9.8, False), (10.1, True)])
def test_late_timer_switch_before_event_requires_independent_check(switch_time, confirmed):
    event = dict(time=10, stone="1-2", seat=0, seats=[0])
    event["recovery"] = dict(method="animation", evidence=dict(motion="incoming"))
    rnd = dict(events=[event])
    mark_weak_indicator(rnd, [timer(9, 0, 23), timer(9.5, 0, 22), timer(switch_time, 1, 23)])
    preserve_player_candidates(rnd)
    assert bool(event.get("timer_confirmed")) is confirmed
    assert event["seats"] == ([0] if confirmed else [0, 1, 2, 3])
    rnd["stone_recovery"] = [dict(method="hand_difference", stone="1-2", seat=1)]
    preserve_player_candidates(rnd)
    assert event["seats"] == [0]


def test_delayed_tile_reading_does_not_use_next_players_countdown():
    # 09-28 20:45, round 3: 6-6 becomes readable after the turn has passed.
    event = dict(
        time=285.92484444444443,
        stone="6-6",
        seat=3,
        seats=[3],
        early_reading_interval=[281.2521, 286.8765333333333],
    )
    rnd = dict(events=[event])
    mark_weak_indicator(
        rnd,
        [
            timer(281.7527111111111, 2, 23),
            timer(282.7707222222222, 2, 22),
            timer(284.83937777777777, 3, 21),
            timer(285.3414333333333, 3, 21),
        ],
    )
    preserve_player_candidates(rnd)
    assert not event.get("timer_confirmed")
    assert event["seats"] == [0, 1, 2, 3]


def test_readable_tile_without_placement_anchor_does_not_prove_actor():
    # Same real round: 5-6 is read at 326.711, next seat already waits at 325.710.
    event = dict(time=326.7106111111111, stone="5-6", seat=2, seats=[2])
    rnd = dict(events=[event])
    mark_weak_indicator(rnd, [timer(325.70964444444445, 2, 23), timer(326.20992222222225, 2, 22)])
    preserve_player_candidates(rnd)
    assert not event.get("timer_confirmed")
    assert event["seats"] == [0, 1, 2, 3]
