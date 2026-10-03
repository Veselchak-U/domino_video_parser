from pathlib import Path

import cv2
import pytest

from domino_video.recognition_plan import OCRFields
from domino_video.vision import ScreenRecognizer, StoneObservation


def test_real_five_tile_reveal_preserves_boundary_stone_and_points():
    image = cv2.imread(str(Path(__file__).parent / "fixtures/reveal_boundary1615.png"))
    obs = ScreenRecognizer().prepare(image, 245.0035, OCRFields()).observation
    assert obs.reveal
    assert set(obs.hands[1]) == {"0-4", "4-5", "2-6", "5-5", "0-6"}
    assert sum(sum(map(int, stone.split("-"))) for stone in obs.hands[1]) == 37


@pytest.mark.parametrize(
    "center,seat",
    [
        (99, None),
        (100, None),
        (126, 1),
        (499, 1),
        (500, 2),
        (1049, 2),
        (1050, 3),
        (1449, 3),
        (1450, None),
    ],
)
def test_reveal_ownership_uses_existing_region_boundaries(monkeypatch, center, seat):
    image = cv2.imread(str(Path(__file__).parent / "fixtures/reveal_boundary1615.png"))
    recognizer = ScreenRecognizer()
    tile = StoneObservation((0, 4), (center - 29, 65, 58, 120))
    monkeypatch.setattr(recognizer, "_stones", lambda *args, **kwargs: [tile])
    obs = recognizer.prepare(image, 245.0035, OCRFields()).observation
    assert obs.hands == [["0-4"] if index == seat else [] for index in range(4)]


def test_boundary_unknown_is_not_lost_or_declared_valid(monkeypatch):
    image = cv2.imread(str(Path(__file__).parent / "fixtures/reveal_boundary1615.png"))
    recognizer = ScreenRecognizer()
    monkeypatch.setattr(
        recognizer,
        "_stones",
        lambda *args, **kwargs: [StoneObservation((None, None), (97, 65, 58, 120))],
    )
    obs = recognizer.prepare(image, 245.0035, OCRFields()).observation
    assert obs.hands[1] == []
    assert not obs.reveal_valid[1]
