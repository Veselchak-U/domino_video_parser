import json
from pathlib import Path

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.validator import GameValidator, InvalidGame


@pytest.mark.parametrize("action", ["left", "right"])
def test_rejects_reversed_non_double(action):
    game = json.loads(Path("contracts/kozel-game.example.json").read_text(encoding="utf-8"))
    # Reverse a legal placement while preserving the stone's identity in the hand.
    move = next(
        move
        for move in game["rounds"][0]["moves"]
        if move["action"] == action and move["stone"][0] != move["stone"][-1]
    )
    move["stone"] = move["stone"][::-1]
    with pytest.raises(InvalidGame, match="край|ориентац"):
        GameValidator().calculate(
            game["teams"], game["rounds"], game["fishVariant"], game["scoreLimit"]
        )


def test_exported_moves_replay_with_exact_orientation(observations):
    reconstructor = GameReconstructor()
    game = reconstructor.build(
        reconstructor.extract(observations),
        next(row.names for row in observations if row.names),
        "withoutEggs",
        50,
    )
    seen = set()
    for rnd in game["rounds"]:
        for move in rnd["moves"]:
            action = move["action"]
            if action == "pass":
                continue
            a, b = map(int, move["stone"].split("-"))
            assert f"{min(a, b)}-{max(a, b)}" in rnd["deal"][move["player"]]
            if action == "start":
                left, right = a, b
            elif action == "left":
                assert b == left
                left = a
            else:
                assert a == right
                right = b
            seen.add(action)
            if a == b:
                seen.add("double")
    assert seen == {"start", "left", "right", "double"}


def test_packaged_schema_matches_contract():
    assert (
        Path("domino_video/schema.json").read_bytes()
        == Path("contracts/kozel-game.schema.json").read_bytes()
    )
