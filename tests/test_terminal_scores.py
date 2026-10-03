from copy import deepcopy
from types import SimpleNamespace

import cv2
import pytest

from domino_video.manager import ParseManager
from domino_video.recognition_plan import OCRFields, score_transition_windows
from domino_video.reconstruct import ReconstructionError, ScoreRecognitionError
from domino_video.vision import Observation, ScreenRecognizer

BEFORE = 489.86938888888886
FINAL = 490.1704
COVERED = 490.38381111111113


def test_real_short_score_is_read_before_grouping_and_checked_against_rules():
    # Record_2026-09-29-14-31-39, SHA256
    # e4c95c8dd07c7b110f8aa4ffcd3872be18c87ad8123bb13e7b6f306b5ff56f89.
    images = {
        t: cv2.imread(f"tests/fixtures/terminal_scores/{name}.png")
        for t, name in [(BEFORE, "before"), (FINAL, "final"), (COVERED, "covered")]
    }
    assert all(image is not None for image in images.values())

    class Reader:
        def interval_frames(self, path, windows):
            for t, image in images.items():
                if any(start <= t <= stop for start, stop in windows):
                    yield t, image

        def selected_frames(self, path, requests):
            for t in sorted(requests):
                yield t, images[t]

    recognizer = ScreenRecognizer()
    observations = [
        recognizer.prepare(images[t], t, OCRFields()).observation for t in (BEFORE, COVERED)
    ]
    rnd = dict(start=450, end=484, complete=True, events=[])
    manager = ParseManager(Reader(), recognizer)
    manager._read_fields(
        "fixture.mp4",
        [rnd],
        observations,
        1,
        SimpleNamespace(message=lambda _: None),
        "cpu",
        "auto",
    )
    final = next((o for o in observations if o.time == FINAL), None)
    assert final is not None
    assert final.scores == (71, 28)
    assert {attempt["field"] for attempt in final.ocr_attempts} == {"score"}
    assert all(attempt["time"] == FINAL for attempt in final.ocr_attempts)
    assert final.names is None and final.counts is None and final.reveal_points is None
    following = dict(start=500, end=510, complete=True, events=[])
    assert manager._groups([rnd, following], observations, 50) == [[rnd], [following]]
    game = dict(
        teams=[dict(name="A"), dict(name="B")],
        rounds=[dict(result=dict(total_score={"A": 71, "B": 28}))],
    )
    manager._check_scores(game, [rnd], observations, 50)
    game["rounds"][0]["result"]["total_score"]["A"] = 72
    with pytest.raises(ReconstructionError, match="на табло"):
        manager._check_scores(game, [rnd], observations, 50)


def observation(time, visible=True):
    return Observation(time, [], [[], [], [], []], None, False, visible, selective=True)


def test_real_result_animation_does_not_displace_readable_ordinary_score():
    # Record_2026-09-28-13-58-19: final cards at 286.8766 can still be
    # classified as an empty supported table during their transition.
    score_times = [285.5769, 286.07932222222223, 286.59265555555555]
    native_times = [286.8766, 286.8922333333333, 286.9087888888889]
    readable = cv2.imread("tests/fixtures/terminal_scores/ordinary_96_29.png")
    cards = cv2.imread("tests/fixtures/terminal_scores/early_results.png")
    assert readable is not None and cards is not None
    images = {
        t: cv2.imread(f"tests/fixtures/terminal_scores/{t:.6f}.png")
        for t in score_times[:-1] + native_times[1:]
    }
    images[score_times[-1]], images[native_times[0]] = readable, cards
    assert all(image is not None for image in images.values())
    calls = []

    class Reader:
        def interval_frames(self, path, windows):
            for t in native_times:
                if any(start <= t < stop for start, stop in windows):
                    yield t, images[t]

        def selected_frames(self, path, requests):
            calls.extend(requests)
            for t in sorted(requests):
                yield t, images[t]

    recognizer = ScreenRecognizer()
    ordinary = [recognizer.prepare(images[t], t, OCRFields()).observation for t in score_times]
    ordinary.append(observation(287.0961222222222, False))
    rnd = dict(start=208.20955555555557, end=274.41192222222224, complete=True, events=[])
    manager = ParseManager(Reader(), recognizer)
    manager._read_fields(
        "fixture.mp4", [rnd], ordinary, 1, SimpleNamespace(message=lambda _: None), "cpu", "auto"
    )
    assert set(score_times).issubset(calls)
    assert next(o for o in ordinary if o.time == score_times[-1]).scores == (96, 29)
    assert all(o.scores is None for o in ordinary if o.time in native_times)
    game = dict(
        teams=[dict(name="A"), dict(name="B")],
        rounds=[dict(result=dict(total_score={"A": 96, "B": 29}))],
    )
    manager._check_scores(game, [rnd], ordinary, 50)


@pytest.mark.parametrize(
    "complete,next_start,expected",
    [(True, 20, [(11, 12)]), (True, 11.5, [(11, 11.5)]), (False, 20, [])],
)
def test_transition_windows_respect_round_boundaries(complete, next_start, expected):
    rounds = [dict(end=10, complete=complete), dict(start=next_start, end=30, complete=True)]
    ordinary = [observation(11), observation(12, False)]
    assert score_transition_windows(rounds, ordinary) == expected
    ordinary[-1].supported = True
    assert score_transition_windows(rounds, ordinary) == []


@pytest.mark.parametrize("native_visible", [True, False])
def test_each_transition_keeps_its_own_bounded_score_requests(native_visible):
    original = [observation(11), observation(12, False), observation(13), observation(14, False)]
    native = [
        observation(t, native_visible) for t in [11.1, 11.2, 11.3, 11.4, 13.1, 13.2, 13.3, 13.4]
    ]
    all_rows = {o.time: o for o in original + native}
    calls = []

    class Reader:
        def interval_frames(self, path, windows):
            for obs in native:
                if any(start <= obs.time < stop for start, stop in windows):
                    yield obs.time, obs

        def selected_frames(self, path, requests):
            for timestamp in sorted(requests):
                yield timestamp, all_rows[timestamp]

    class Recognizer:
        def observe(self, image, timestamp, fields):
            obs = deepcopy(image)
            if fields.scores:
                calls.append((timestamp, fields))
                # The later apparently empty UI cannot read any scoreboard.
                obs.scores = (71, 28) if 11 < timestamp < 12 else None
                obs.limit = 50 if obs.scores else None
            return obs

    manager = ParseManager(Reader(), Recognizer())
    rnd = dict(
        start=1,
        end=10,
        complete=True,
        events=[],
        remaining=[[], [], [], []],
        remaining_confirmed=[False] * 4,
    )
    manager._read_fields(
        "fixture.mp4", [rnd], original, 1, SimpleNamespace(message=lambda _: None), "cpu", "auto"
    )
    native_calls = [(t, fields) for t, fields in calls if t not in [11, 12, 13, 14]]
    expected = [11.2, 11.3, 11.4, 13.2, 13.3, 13.4] if native_visible else []
    assert [t for t, _ in native_calls] == expected
    assert all(fields == OCRFields(scores=True) for _, fields in native_calls)
    assert len({o.time for o in original}) == len(original)
    assert rnd["remaining_confirmed"] == [False] * 4
    assert all(o.names is None and o.counts is None and o.reveal_points is None for o in original)
    if native_visible:
        following = dict(start=20, end=30, complete=True)
        assert manager._groups([rnd, following], original, 50) == [[rnd], [following]]
    else:
        assert not any(o.scores for o in original)
        with pytest.raises(ScoreRecognitionError):
            manager._check_scores(dict(rounds=[{}]), [rnd], original, 50)
