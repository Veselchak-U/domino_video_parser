import json
from copy import deepcopy
from pathlib import Path

import cv2
import pytest

from domino_video.recognition_plan import OCRFields, recognition_requests
from domino_video.remaining_evidence import confirm_empty_remaining
from domino_video.vision import Observation, ScreenRecognizer, StoneObservation


def test_real_reveal_reads_points_including_zero_with_actual_ocr():
    recognizer = ScreenRecognizer()
    try:
        image = cv2.imread("tests/fixtures/reveal_zero_points.png")
        observation = recognizer.observe(image, 224.50272222222222, True)
        assert getattr(observation, "reveal_points", None) == (23, 15, 4, 0)
        assert observation.reveal_valid[3]
    finally:
        recognizer.close()


def test_small_real_animation_badge_does_not_read_as_final_zero():
    image = cv2.imread("tests/fixtures/reveal_partial_points.png")
    prepared = ScreenRecognizer().prepare(image, 224.0008777777778, OCRFields(reveal_points=True))
    assert prepared.crops["reveal_points"][3] is None


@pytest.mark.parametrize("size", [(60, 120), (120, 60)])
def test_unreadable_hand_contour_invalidates_empty_region(monkeypatch, size):
    recognizer = ScreenRecognizer()
    monkeypatch.setattr(
        recognizer, "_stones", lambda *args: [StoneObservation((None, None), (1250, 65, *size))]
    )
    prepared = recognizer.prepare(
        cv2.imread("tests/fixtures/reveal_zero_points.png"), 224.5, OCRFields()
    )
    assert prepared.observation.hands[3] == []
    assert prepared.observation.reveal_valid[3] is False
    assert prepared.crops == {}


def evidence():
    rows = []
    for time in [224.50272222222222, 225.0033]:
        row = Observation(time, [], [["6-6"], ["1-5"], ["1-2"], []], None, True, True)
        row.reveal_valid = (True,) * 4
        row.reveal_panels_complete = (True,) * 4
        row.reveal_points = (12, 6, 3, 0)
        rows.append(row)
    rnd = dict(
        start=200,
        end=222.21516666666668,
        complete=True,
        remaining=[["6-6"], ["1-5"], ["1-2"], []],
        remaining_confirmed=[True, True, True, False],
        last_active=1,
    )
    return rnd, rows


def test_repeated_zero_points_confirms_only_empty_unknown_hand():
    rnd, rows = evidence()
    original = deepcopy(rows)
    confirm_empty_remaining([rnd], rows)
    assert rnd["remaining_confirmed"] == [True] * 4
    assert rnd["remaining_evidence"] == [
        dict(seat=4, method="zero_reveal_points", times=[o.time for o in rows], points=[0, 0])
    ]
    assert rows == original
    confirm_empty_remaining([rnd], rows)
    assert len(rnd["remaining_evidence"]) == 1


@pytest.mark.parametrize(
    "kind",
    [
        "single",
        "duplicate_time",
        "invalid",
        "nonzero",
        "unknown",
        "nonempty",
        "zero_tile",
        "not_reveal",
    ],
)
def test_incomplete_or_contradictory_evidence_cannot_prove_empty(kind):
    rnd, rows = evidence()
    if kind == "single":
        rows.pop()
    elif kind == "duplicate_time":
        rows[1].time = rows[0].time
    elif kind == "invalid":
        rows[1].reveal_valid = (True, True, True, False)
    elif kind == "nonzero":
        rows[1].reveal_points = (12, 6, 3, 1)
    elif kind == "unknown":
        rows[1].reveal_points = (12, 6, 3, None)
    elif kind == "nonempty":
        rows[0].hands[3] = ["0-3"]
    elif kind == "zero_tile":
        rows[0].hands[3] = ["0-0"]
    else:
        rows[1].reveal = False
    confirm_empty_remaining([rnd], rows)
    assert rnd["remaining_confirmed"][3] is False
    assert not rnd.get("remaining_evidence")


def test_final_point_requests_are_bounded_and_only_for_unknown_hands():
    rnd, rows = evidence()
    rnd["events"] = []
    rows += [deepcopy(rows[-1]) for _ in range(7)]
    for i, row in enumerate(rows):
        row.time = 224 + i / 2
    requests = recognition_requests([rnd], rows)
    assert [t for t, fields in requests.items() if fields.reveal_points] == [226.5, 227, 227.5, 228]
    rnd["remaining_confirmed"] = [True] * 4
    assert not recognition_requests([rnd], rows)


def test_manager_transfers_point_evidence_and_confirms_after_reading(monkeypatch):
    from domino_video.manager import ParseManager
    from domino_video.pipeline import ObservationPipeline

    rnd, completed = evidence()
    rnd["events"] = []
    observations = deepcopy(completed)
    for obs in observations:
        obs.reveal_points = None

    class Reader:
        def selected_frames(self, path, timestamps):
            for time in timestamps:
                yield time, None

    class Progress:
        def message(self, text):
            pass

    def observe(self, frames):
        assert list(frames) == [(o.time, None) for o in completed]
        yield from completed

    monkeypatch.setattr(ObservationPipeline, "observe", observe)
    manager = ParseManager(reader=Reader())
    manager._read_fields("source.mp4", [rnd], observations, 1, Progress(), "cpu", 1)
    assert rnd["remaining_confirmed"][3]
    assert observations[0].reveal_points == (12, 6, 3, 0)
    assert rnd["remaining_evidence"][0]["times"] == [o.time for o in completed]


def test_real_next_deal_cannot_displace_completed_point_panels():
    root = Path("tests/fixtures/reveal_selection")
    frames = json.loads((root / "source.json").read_text(encoding="utf8"))["frames"]
    recognizer = ScreenRecognizer()
    try:
        rows = [
            recognizer.prepare(cv2.imread(str(root / frame["file"])), frame["time"], OCRFields())
            for frame in frames
        ]
        assert all(not row.crops for row in rows)
        rnd = dict(
            start=0,
            end=144.396956,
            complete=True,
            events=[],
            remaining=[[], ["0-2"], ["0-6"], ["4-4"]],
            remaining_confirmed=[False, True, True, True],
        )
        requests = recognition_requests([rnd], [row.observation for row in rows])
        selected = [time for time, fields in requests.items() if fields.reveal_points]
        assert selected == [frame["time"] for frame in frames[:4]]
        completed = [
            recognizer.observe(
                cv2.imread(str(root / frame["file"])), frame["time"], requests[frame["time"]]
            )
            for frame in frames[:4]
        ]
        assert [row.reveal_points[0] for row in completed] == [0, None, 0, None]
        confirm_empty_remaining([rnd], completed)
        assert rnd["remaining_confirmed"] == [True] * 4
        assert rnd["remaining_evidence"][0]["times"] == [frames[i]["time"] for i in (0, 2)]
    finally:
        recognizer.close()


@pytest.mark.parametrize("panels", [None, (False,) * 4, (True, True, True, False)])
def test_unknown_or_other_players_panel_cannot_request_empty_hand(panels):
    rnd, rows = evidence()
    rnd["events"] = []
    for row in rows:
        row.reveal_panels_complete = panels
    assert not recognition_requests([rnd], rows)
