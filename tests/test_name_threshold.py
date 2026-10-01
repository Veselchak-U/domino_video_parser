import cv2
import pytest

from domino_video.name_ocr import NameOCR, NameReading, NameSymbol
from domino_video.ocr_result import OCRLine, OCRResult
from domino_video.player_names import PlayerNameResolver
from domino_video.vision import ScreenRecognizer


@pytest.mark.parametrize(
    "confidence,accepted", [(0.699, False), (0.7, False), (0.701, True), (0.75, True)]
)
def test_name_threshold_boundary_and_numeric_fields_are_independent(confidence, accepted):
    reading = NameReading(symbols=(NameSymbol("A", confidence, 0, 10),))
    assert reading.text == ("A" if accepted else "*")
    prepared = ScreenRecognizer().prepare(cv2.imread("tests/fixtures/frame_20.png"), 20, True)
    obs = prepared.finish(lambda crop: OCRResult((OCRLine("6", 0.75),)), lambda crop: reading)
    assert all(a["threshold"] == (0.7 if a["field"] == "name" else 0.8) for a in obs.ocr_attempts)
    assert obs.counts[1:] == (None, None, None) and obs.scores is None
    resolved = PlayerNameResolver().resolve([obs], [dict(start=20, end=100)])
    assert resolved.names[0] == ("A" if accepted else "Unrecognized_1")
    assert all(e["threshold"] == 0.7 for e in resolved.entries)


def test_supplementary_name_symbol_at_75_percent_can_fill_uncertain_primary():
    reader = object.__new__(NameOCR)
    assert (
        NameReading(
            symbols=reader.merge((NameSymbol("?", 0.2, 0, 10),), (NameSymbol("@", 0.75, 0, 10),))
        ).text
        == "@"
    )
    assert (
        NameReading(
            symbols=reader.merge((NameSymbol("М", 0.75, 0, 10),), (NameSymbol("M", 0.99, 0, 10),))
        ).text
        == "М"
    )


@pytest.mark.parametrize("confidence,accepted", [(0.75, False), (0.8, False), (0.801, True)])
def test_numeric_line_threshold_stays_at_eighty_percent(confidence, accepted):
    assert bool(OCRResult((OCRLine("6", confidence),)).text) == accepted
