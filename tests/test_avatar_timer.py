import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from domino_video.avatar_timer import AvatarTimerReader
from domino_video.screen_profile import ScreenProfile
from domino_video.vision import ScreenRecognizer

ROOT = Path(__file__).parent / "fixtures" / "avatar_timer"
CASES = json.loads((ROOT / "manifest.json").read_text(encoding="utf8"))


def frame(case):
    image = np.zeros((720, 1608, 3), np.uint8)
    strips = cv2.imread(str(ROOT / case["file"]))
    for seat, (x, y) in enumerate(ScreenProfile.avatar_centers):
        image[y - 25 : y + 25, x - 37 : x + 37] = strips[:, seat * 74 : (seat + 1) * 74]
    return image


@pytest.fixture(scope="module")
def reader():
    return AvatarTimerReader()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["file"])
def test_real_countdown_and_disappearing_digits(case, reader):
    reading = reader.read(frame(case))
    if case["expected"] is None:
        assert reading.status == "absent"
        assert reading.seat is None
    else:
        assert reading.status == "confirmed"
        assert (reading.seat, reading.value) == tuple(case["expected"])


def test_two_timers_remain_ambiguous(reader):
    image = frame(CASES[0])
    x, y = ScreenProfile.avatar_centers[3]
    a, b = ScreenProfile.avatar_centers[0]
    image[b - 25 : b + 25, a - 37 : a + 37] = image[y - 25 : y + 25, x - 37 : x + 37]
    reading = reader.read(image)
    assert reading.status == "ambiguous"
    assert reading.seat is None
    assert reading.value is None


@pytest.mark.parametrize("turns", range(4))
def test_scaled_rotated_real_frame_after_normalization(turns, reader):
    source = cv2.imread(str(ROOT / "scaled_real_frame.jpg"))
    normalized = ScreenRecognizer().normalize(np.rot90(source, turns).copy())
    reading = reader.read(normalized)
    assert (reading.status, reading.seat, reading.value) == ("confirmed", 3, 21)


@pytest.mark.parametrize(
    "text,confidence,status,value",
    [
        ("0", 0.95, "confirmed", 0),
        ("5", 0.95, "confirmed", 5),
        ("8", 0.95, "confirmed", 8),
        ("30", 0.95, "confirmed", 30),
        ("31", 0.95, "ambiguous", None),
        ("23", 0.5, "ambiguous", None),
        ("face", 0.99, "ambiguous", None),
    ],
)
def test_numeric_contract_separately_from_real_ocr(monkeypatch, text, confidence, status, value):
    import rapidocr_onnxruntime

    monkeypatch.setattr(
        rapidocr_onnxruntime,
        "RapidOCR",
        lambda **kw: lambda *args, **kwargs: ([[text, confidence]], None),
    )
    reading = AvatarTimerReader().read(frame(CASES[0]))
    assert (reading.status, reading.value) == (status, value)


def test_real_negative_never_initializes_ocr(monkeypatch):
    import rapidocr_onnxruntime

    def forbidden(**kwargs):
        raise AssertionError("OCR must not run for decorations or a disappearing timer")

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", forbidden)
    detector = AvatarTimerReader()
    for case in CASES:
        if case["expected"] is None:
            assert detector.read(frame(case)).status == "absent"
