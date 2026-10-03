import gzip
import json
from pathlib import Path

import cv2

from domino_video.recognition_plan import OCRFields
from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation, ScreenRecognizer, StoneObservation

FIXTURES = Path("tests/fixtures/anchor_recovery")


def native_rows():
    with gzip.open(FIXTURES / "observations.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    result = []
    for row in fixture["early"] + fixture["late"]:
        for key in ("board", "uncertain_board"):
            row[key] = [StoneObservation(**tile) for tile in row[key]]
        result.append(Observation(**row))
    return result


def test_real_native_upper_unknown_is_retained_without_reading_hidden_pips():
    image = cv2.imread(str(FIXTURES / "early.png"))
    recognizer = ScreenRecognizer()
    obs = recognizer.prepare(image, 114.30175555555556, OCRFields(), read_motion=True).observation
    assert any(tile.box == (1192, 137, 136, 66) for tile in obs.uncertain_board)
    assert "4-5" not in {tile.stone for tile in obs.board}
    ordinary = recognizer.prepare(image, obs.time, OCRFields()).observation
    assert not any(tile.box == (1192, 137, 136, 66) for tile in ordinary.uncertain_board)


def test_real_two_anchor_identity_refines_existing_late_event():
    rnd = json.loads((FIXTURES / "round.json").read_text(encoding="utf8"))
    GameReconstructor().integrate_recovery([rnd], native_rows())
    events = [event for event in rnd["events"] if event["stone"] == "4-5"]
    assert len(events) == 1
    assert events[0]["time"] < 114.4
    assert events[0]["recovery"]["method"] == "anchor_similarity"
