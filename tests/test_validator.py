"""Право следующего захода после реального выхода при нулевой записи."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from domino_video.validator import GameValidator, InvalidGame


def recording():
    return json.loads(Path("tests/fixtures/zero_score_exit.json").read_text(encoding="utf8"))


def test_real_empty_hand_at_zero_keeps_finisher_as_next_starter():
    fixture = recording()
    game = GameValidator().calculate(fixture["teams"], fixture["rounds"], "withoutEggs", 50)
    assert game["rounds"][0]["result"]["reason"] == "emptyHand"
    assert game["rounds"][0]["result"]["total_score"] == {"Team A": 0, "Team B": 0}
    assert game["rounds"][0]["result"]["fence"] == {"Team A": 10}
    first, second = game["rounds"][:2]
    assert second["moves"][0] == dict(
        player=first["result"]["finished_player"], action="start", stone="0-2"
    )
    assert game["rounds"][-1]["result"]["total_score"] == {"Team A": 59, "Team B": 61}
    assert GameValidator().validate(game) == game


def test_zero_exit_starter_also_applies_with_eggs():
    fixture = recording()
    # These two real rounds precede the fish whose scoring depends on the variant.
    # They must replay fully; only the missing end of the whole match is rejected.
    with pytest.raises(InvalidGame, match="Партия не завершена"):
        GameValidator().calculate(fixture["teams"], fixture["rounds"][:2], "withEggs", 50)


@pytest.mark.parametrize("kind", ["first_stone", "second_player", "repeat_start"])
def test_zero_exit_does_not_relax_first_start_or_turn(kind):
    fixture = recording()
    rounds = fixture["rounds"]
    if kind == "first_stone":
        rounds[0]["moves"][0]["stone"] = "0-2"
    elif kind == "second_player":
        rounds[1]["moves"][0]["player"] = fixture["teams"][0]["players"][0]["name"]
    else:
        rounds[1]["moves"][1]["action"] = "start"
    with pytest.raises(InvalidGame):
        GameValidator().calculate(fixture["teams"], rounds, "withoutEggs", 50)


def test_input_reason_cannot_change_calculated_starter():
    fixture = recording()
    fixture["rounds"][0]["result"] = dict(reason="eggs", finished_player="invented")
    game = GameValidator().calculate(fixture["teams"], fixture["rounds"], "withoutEggs", 50)
    assert game["rounds"][0]["result"]["reason"] == "emptyHand"


def test_zero_eggs_still_requires_new_holder_of_double():
    game = json.loads(Path("contracts/kozel-game.example.json").read_text(encoding="utf8"))
    assert GameValidator().validate(game) == game
    changed = deepcopy(game)
    changed["rounds"][1]["moves"][0]["player"] = game["rounds"][0]["result"]["finished_player"]
    with pytest.raises(InvalidGame, match="очередь"):
        GameValidator().validate(changed)
