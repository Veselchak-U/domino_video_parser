import gzip
import json

import pytest


@pytest.fixture
def observations():
    from domino_video.vision import Observation, StoneObservation

    with gzip.open("tests/fixtures/observations.json.gz", "rt", encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        row["board"] = [StoneObservation(tuple(s["values"]), tuple(s["box"])) for s in row["board"]]
        if row["scores"] is not None:
            row["scores"] = tuple(row["scores"])
    return [Observation(**row) for row in rows]


@pytest.fixture
def sample_game():
    with open("tests/fixtures/sample_game.json", encoding="utf-8") as f:
        return json.load(f)
