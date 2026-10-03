import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from domino_video.recognition_plan import OCRFields
from domino_video.vision import ScreenRecognizer

FIXTURES = Path("tests/fixtures/active_indicator")
CASES = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf8"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["file"])
def test_real_indicator_separates_timer_from_portrait_and_medal(case):
    prepared = ScreenRecognizer().prepare(
        cv2.imread(str(FIXTURES / case["file"])), case["time"], OCRFields()
    )
    # These real transition frames retain the old rim after their digits vanish.
    absent = case["file"] in {
        "118.1.png",
        "118.25.png",
        "120.25.png",
        "120.5.png",
        "1358-29.819833.png",
    }
    assert prepared.observation.active == (None if absent else case["active"])
    assert not prepared.crops


def test_decorated_timer_preserves_existing_detection():
    assert (
        ScreenRecognizer().prepare(cv2.imread("tests/fixtures/frame_20.png"), 20).observation.active
        == 3
    )


def paint_arcs(image, seats):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for seat in seats:
        center = [(188, 538), (157, 80), (691, 80), (1224, 80)][seat]
        cv2.ellipse(hsv, center, (53, 53), 0, 100, 280, (20, 255, 255), 9)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_unique_digits_override_two_decorative_arcs():
    image = ScreenRecognizer().normalize(cv2.imread("tests/fixtures/frame_20.png"))
    obs = ScreenRecognizer().prepare(paint_arcs(image, [0, 1]), 20).observation
    assert obs.active == 3
    assert obs.active_method == "timer"


def test_no_timer_is_unknown():
    image = np.full((720, 1608, 3), (55, 150, 140), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    assert ScreenRecognizer().prepare(image, 20).observation.active is None


def test_reveal_never_has_an_active_player_even_with_a_synthetic_arc():
    image = ScreenRecognizer().normalize(cv2.imread("tests/fixtures/frame_84.png"))
    obs = ScreenRecognizer().prepare(paint_arcs(image, [0]), 84).observation
    assert obs.reveal
    assert obs.active is None


@pytest.mark.parametrize("turns,scale", [(1, 1), (2, 0.75), (3, 0.75)])
def test_real_arc_survives_input_rotation_and_scaling(turns, scale):
    image = cv2.imread(str(FIXTURES / "117.95.png"))
    image = cv2.resize(np.rot90(image, turns), None, fx=scale, fy=scale)
    assert ScreenRecognizer().prepare(image, 117.95).observation.active == 3


def test_ambiguous_frame_does_not_add_a_false_player_to_history():
    from domino_video.reconstruct import GameReconstructor
    from domino_video.vision import Observation, StoneObservation

    image = cv2.imread(str(FIXTURES / "1358-29.819833.png"))
    ambiguous = ScreenRecognizer().prepare(paint_arcs(image, [0, 1]), 20).observation.active
    assert ambiguous is None
    stone = StoneObservation((1, 1), (800, 280, 60, 120))
    rows = [Observation(1, [], [[], [], [], []], 3, False, True)]
    rows.extend(
        Observation(t, [stone], [[], [], [], []], ambiguous, False, True) for t in [2, 2.5, 3]
    )
    event = GameReconstructor().extract(rows)[0]["events"][0]
    assert event["seats"] == [3]
