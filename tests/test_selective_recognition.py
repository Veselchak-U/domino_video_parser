from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from domino_video.player_names import PlayerNameResolver
from domino_video.recognition_plan import OCRFields, recognition_requests
from domino_video.video import VideoReader
from domino_video.vision import Observation, ScreenRecognizer


def test_default_stone_sampling_is_two_frames_per_second(monkeypatch):
    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(thread_count=0)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, **kwargs):
            for t in [0, 0.25, 0.5, 0.75, 1]:
                yield SimpleNamespace(time=t, to_ndarray=lambda **kwargs: None)

    monkeypatch.setattr("domino_video.video.av.open", lambda path: Container())
    assert [t for t, _ in VideoReader().frames("fixture")] == [0, 0.5, 1]


@pytest.mark.parametrize("turns,scale", [(0, 1), (1, 0.75), (2, 1), (3, 0.75)])
def test_result_table_has_names_mapped_to_seats(turns, scale):
    image = cv2.imread("tests/fixtures/frame_results.png")
    image = cv2.resize(np.rot90(image, turns), None, fx=scale, fy=scale)
    prepared = ScreenRecognizer().prepare(image, 287, True)
    assert getattr(prepared.observation, "result_table", False)
    assert len(prepared.crops["names"]) == 4
    assert "counts" not in prepared.crops and "scores" not in prepared.crops


def test_names_can_be_read_after_last_round_without_leaking_next_game():
    rows = [
        Observation(t, [], [[], [], [], []], None, False, False, names=[name] * 4)
        for t, name in [(20, "Wrong"), (110, "Final"), (130, "Next")]
    ]
    resolved = PlayerNameResolver().resolve(rows, [dict(start=10, end=100, names_end=120)])
    assert resolved.names == ["Final", "Final_2", "Final_3", "Final_4"]


def test_field_requests_do_not_read_unrequested_names_or_counts():
    from domino_video.ocr_result import OCRResult

    image = cv2.imread("tests/fixtures/frame_20.png")
    recognizer = ScreenRecognizer()
    assert recognizer.prepare(image, 20, OCRFields()).crops == {}
    counts = recognizer.prepare(image, 20, OCRFields(counts=True))
    assert set(counts.crops) == {"counts"}
    obs = counts.finish(lambda crop: OCRResult.from_rows([(None, "6", 0.99)]))
    assert obs.counts == (6, 6, 6, 6) and obs.names is None and obs.scores is None
    scores = recognizer.prepare(image, 20, OCRFields(scores=True))
    assert set(scores.crops) == {"scores"}


def test_only_ambiguous_moves_request_counters_and_final_fields_are_bounded():
    def row(t, board=True, table=False):
        return Observation(
            t,
            [object()] if board else [],
            [[], [], [], []],
            0,
            False,
            not table,
            result_table=table,
        )

    rows = [row(t) for t in [2, 2.5, 3, 5, 5.5, 6]]
    rows += [row(t, False) for t in [10.5, 11, 11.5, 12]]
    rows += [row(t, False, True) for t in [13, 13.5, 14, 14.5]]
    rnd = dict(
        start=1,
        end=10,
        complete=True,
        events=[
            dict(time=1, seats=[0], stone="1-1"),
            dict(time=4, seats=[0, 1], stone="1-6"),
        ],
    )
    requests = recognition_requests([rnd], rows)
    assert [t for t, f in requests.items() if f.counts] == [5, 5.5]
    assert [t for t, f in requests.items() if f.scores] == [11, 11.5, 12]
    assert [t for t, f in requests.items() if f.names] == [13, 13.5, 14]


def test_delayed_final_table_is_read_without_requesting_gameplay_names():
    rnd = dict(start=1, end=10, complete=True, events=[])
    rows = [Observation(30, [], [[], [], [], []], None, False, False, result_table=True)]
    assert recognition_requests([rnd], rows) == {30: OCRFields(names=True)}


def test_late_score_before_next_round_remains_available_for_validation():
    rounds = [
        dict(start=1, end=10, complete=True, events=[]),
        dict(start=30, end=None, complete=False, events=[]),
    ]
    rows = [Observation(27, [], [[], [], [], []], None, False, True)]
    assert recognition_requests(rounds, rows) == {27: OCRFields(scores=True)}
