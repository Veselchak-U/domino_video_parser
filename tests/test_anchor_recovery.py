import gzip
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2

from domino_video.recognition_plan import OCRFields
from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
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
    saved = deepcopy(rnd)
    GameReconstructor().integrate_recovery([rnd], native_rows())
    assert rnd == saved


def test_two_separate_unknown_slots_cannot_claim_one_late_event():
    rnd = json.loads((FIXTURES / "round.json").read_text(encoding="utf8"))
    rows = native_rows()
    extra = [replace(row, time=row.time + 0.5) for row in rows if row.time < 115]
    before = deepcopy(rnd)
    StoneRecovery()._anchor_placements(rnd, sorted(rows + extra, key=lambda row: row.time))
    assert rnd == before


def test_independent_incoming_proof_is_not_replaced_by_anchor_identity():
    rnd = json.loads((FIXTURES / "round.json").read_text(encoding="utf8"))
    event = next(event for event in rnd["events"] if event["stone"] == "4-5")
    event["recovery"] = dict(method="animation", evidence=dict(motion="incoming"))
    before = deepcopy(event)
    StoneRecovery()._anchor_placements(rnd, native_rows())
    assert event == before


def test_unknown_outside_delayed_reading_windows_does_not_reorder_events():
    rnd = json.loads((FIXTURES / "round.json").read_text(encoding="utf8"))
    event = next(event for event in rnd["events"] if event["stone"] == "4-5")
    event.pop("early_reading_interval", None)
    rnd["events"] = [event]
    before = deepcopy(event)
    StoneRecovery()._anchor_placements(rnd, native_rows())
    assert event == before


def test_full_real_first_round_recovers_hidden_tile_before_next_placement():
    with gzip.open(FIXTURES / "full_first_round.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    rows = []
    for row in fixture["observations"]:
        for key in ("board", "uncertain_board"):
            row[key] = [StoneObservation(**tile) for tile in row[key]]
        rows.append(Observation(**row))
    rnd = fixture["round"]
    GameReconstructor().integrate_recovery([rnd], rows)
    event = next(e for e in rnd["events"] if e["stone"] == "4-5")
    assert event["time"] < 115
    assert event["recovery"]["method"] == "anchor_similarity"
