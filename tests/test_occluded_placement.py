import gzip
import json
from copy import deepcopy

from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation, StoneObservation


def recording():
    with gzip.open(
        "tests/fixtures/occluded_placement1341.json.gz", "rt", encoding="utf8"
    ) as stream:
        data = json.load(stream)
    for row in data["observations"]:
        for key in ("board", "uncertain_board"):
            row[key] = [StoneObservation(**tile) for tile in row[key]]
    return data["round"], [Observation(**row) for row in data["observations"]]


def test_real_readable_flight_precedes_next_move_despite_hidden_installation():
    rnd, rows = recording()
    GameReconstructor().integrate_recovery([rnd], rows)
    event = next(e for e in rnd["events"] if e["stone"] == "2-5")
    assert event["time"] < 87.6
    assert event["recovery"]["method"] == "occluded_placement"
    assert event["recovery"]["evidence"]["neighbor"] == "1-2"
    before = deepcopy(rnd)
    GameReconstructor().integrate_recovery([rnd], rows)
    assert rnd == before
