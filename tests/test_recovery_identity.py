import gzip
import json

import pytest
from test_stone_recovery import load_recording_rows

from domino_video.reconstruct import GameReconstructor
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


def test_existing_tile_misread_does_not_redate_later_real_placement():
    with gzip.open("tests/fixtures/old_tile_identity.json.gz", "rt", encoding="utf8") as stream:
        fixture = json.load(stream)
    rows = []
    for value in fixture["observations"]:
        for field in ["board", "uncertain_board"]:
            value[field] = [
                StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in value[field]
            ]
        rows.append(Observation(**value))
    rnd = fixture["round"]
    StoneRecovery().integrate([rnd], rows)
    event = next(e for e in rnd["events"] if e["stone"] == "2-3")
    assert event["time"] == pytest.approx(76.99874444444444)
    assert not event.get("recovery")
    assert any(e["region"] == [225, 454, 120, 59] for e in rnd["unresolved"])


def test_actual_native_placement_still_resolves_once():
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(
        load_recording_rows("tests/fixtures/occluded_observations.json.gz")
    )
    reconstructor.integrate_recovery(
        rounds, load_recording_rows("tests/fixtures/occluded_motion_observations.json.gz")
    )
    moves = [e for e in rounds[0]["events"] if e["stone"] == "0-1"]
    assert len(moves) == 1
    assert moves[0]["recovery"]["method"] == "animation"
    assert moves[0]["time"] < rounds[0]["end"]
