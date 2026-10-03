from domino_video.round_start import confirm_round_starts, start_requests, start_windows
from domino_video.vision import Observation


def row(time, counts=(7, 7, 7, 7), hand=None):
    obs = Observation(time, [], [hand or [f"0-{i}" for i in range(7)], [], [], []], 2, False, True)
    obs.counts = counts
    return obs


def test_full_deal_confirms_start_and_preserves_observed_hand():
    rnd = dict(start=10, end=30, events=[dict(time=10)], start_observed=False)
    observations = [row(9.4), row(9.6)]
    confirm_round_starts([rnd], observations)
    assert rnd["start_observed"]
    assert rnd["initial_hand"] == [f"0-{i}" for i in range(7)]
    assert rnd["start_evidence"]["times"] == [9.4, 9.6]


def test_six_or_unknown_is_not_a_new_deal():
    rnd = dict(start=10, end=30, events=[dict(time=10)], start_observed=False)
    confirm_round_starts([rnd], [row(9.4, (7, 7, 7, 6)), row(9.6, (7, None, 7, 7))])
    assert not rnd["start_observed"]
    assert "start_evidence" not in rnd


def test_invalid_hand_region_does_not_prove_deal():
    rnd = dict(start=10, end=30, events=[dict(time=10)], start_observed=False)
    rows = [row(9.4), row(9.6)]
    for obs in rows:
        obs.hand_regions_valid = (False, False, False, False)
    confirm_round_starts([rnd], rows)
    assert not rnd["start_observed"]


def test_requests_are_bounded_and_do_not_cross_previous_round():
    rounds = [
        dict(start=1, end=9, events=[dict(time=1)]),
        dict(start=10, end=30, events=[dict(time=10)]),
    ]
    assert start_windows(rounds)[1][0] >= 9
    requests = start_requests(rounds, [row(t / 10) for t in range(85, 101)])
    assert len(requests) <= 3
    assert all(9 < time < 10 for time in requests)


def test_one_frame_or_mismatched_hands_does_not_prove_full_deal():
    rnd = dict(start=10, end=30, events=[dict(time=10)], start_observed=False)
    confirm_round_starts([rnd], [row(9.5)])
    assert not rnd["start_observed"]
    confirm_round_starts([rnd], [row(9.4), row(9.6, hand=[f"1-{i}" for i in range(7)])])
    assert not rnd["start_observed"]


def test_real_short_deal_and_incoming_tile_are_distinguished():
    import cv2

    from domino_video.recognition_plan import OCRFields
    from domino_video.vision import ScreenRecognizer

    recognizer = ScreenRecognizer()
    rows = [
        recognizer.observe(
            cv2.imread(f"tests/fixtures/active_indicator/{time}.png"),
            float(time),
            OCRFields(counts=True),
        )
        for time in ("117.75", "117.95")
    ]
    incoming = recognizer.observe(
        cv2.imread("tests/fixtures/round_start/incoming_first.png"), 118.017, OCRFields(counts=True)
    )
    recognizer.close()
    assert all(o.counts == (7, 7, 7, 7) and not o.board for o in rows)
    assert not incoming.board and incoming.counts == (7, 7, 7, 6)
    rnd = dict(start=118.3, end=125, events=[dict(time=118.3)], start_observed=False)
    confirm_round_starts([rnd], rows + [incoming])
    assert rnd["start_evidence"]["times"] == [117.75, 117.95]
