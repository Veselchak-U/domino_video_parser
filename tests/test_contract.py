import json
from pathlib import Path

import pytest


@pytest.fixture
def example():
    return json.loads(Path("contracts/kozel-game.example.json").read_text(encoding="utf-8"))


def test_accepts_official_completed_game(example):
    from domino_video.validator import GameValidator

    assert GameValidator().validate(example) == example


def test_rejects_changed_score(example):
    from domino_video.validator import GameValidator, InvalidGame

    example["rounds"][0]["result"]["total_score"]["Север"] = 1
    with pytest.raises(InvalidGame):
        GameValidator().validate(example)


@pytest.mark.parametrize("limit,round_count", [(101, 3), (50, 2)])
def test_respects_selected_limit(example, limit, round_count):
    from domino_video.validator import GameValidator

    example["scoreLimit"] = limit
    example["rounds"] = example["rounds"][:round_count]
    assert GameValidator().validate(example) == example


@pytest.mark.parametrize(
    "corruption",
    [
        "extra",
        "duplicate_name",
        "duplicate_seat",
        "same_team",
        "stone",
        "turn",
        "side",
        "pass",
        "start",
        "number",
        "premature",
        "late",
    ],
)
def test_rejects_invalid_game(example, corruption):
    from domino_video.validator import GameValidator, InvalidGame

    first = example["rounds"][0]
    match corruption:
        case "extra":
            example["unexpected"] = True
        case "duplicate_name":
            example["teams"][1]["players"][0]["name"] = "Анна"
        case "duplicate_seat":
            example["teams"][1]["players"][0]["seat"] = 1
        case "same_team":
            example["teams"][0]["players"][1]["seat"] = 2
            example["teams"][1]["players"][0]["seat"] = 3
        case "stone":
            first["deal"]["Анна"][0] = "0-0"
        case "turn":
            first["moves"][1]["player"] = "Анна"
        case "side":
            first["moves"][3]["action"] = "right"
        case "pass":
            first["moves"][1] = {"player": "Борис", "action": "pass"}
        case "start":
            first["moves"][0]["stone"] = "0-4"
        case "number":
            first["number"] = 2
        case "premature":
            example["rounds"] = example["rounds"][:1]
        case "late":
            example["scoreLimit"] = 50
    with pytest.raises(InvalidGame):
        GameValidator().validate(example)


@pytest.mark.parametrize(
    "hand,expected", [([], 0), (["0-0"], 10), (["0-0", "1-2"], 3), (["6-6"], 12)]
)
def test_scores_each_hand_independently(hand, expected):
    from domino_video.rules import ScoringRules

    assert ScoringRules().hand_value(hand) == expected


@pytest.mark.parametrize("variant", ["withEggs", "withoutEggs"])
@pytest.mark.parametrize("limit", [50, 101])
def test_score_limit_boundaries(variant, limit):
    from domino_video.rules import ScoringRules

    rules = ScoringRules()
    for value in [limit - 1, limit, limit + 1]:
        result = rules.game_result({"A": value, "B": 0}, limit)
        assert result == (None if value < limit else {"winner": "B"})


@pytest.mark.parametrize(
    "score,fence,remaining,eggs,new_score,new_fence",
    [
        (0, 0, 12, 28, 0, 40),
        (0, 8, 12, 28, 0, 48),
        (0, 8, 13, 28, 49, 0),
        (20, 0, 12, 28, 60, 0),
        (0, 96, 12, 28, 0, 136),
    ],
)
def test_opens_record_only_from_own_penalty(score, fence, remaining, eggs, new_score, new_fence):
    from domino_video.rules import ScoringRules

    result = ScoringRules().settle(
        "withEggs",
        "emptyHand",
        "B",
        {"A": remaining, "B": 0},
        {"A": score, "B": 0},
        {"A": fence, "B": 8},
        eggs,
    )
    assert result["total_score"] == {"A": new_score, "B": 0}
    assert result.get("fence", {}) == ({"A": new_fence} if new_fence else {})
    assert result["round_score"] == {"A": remaining + eggs, "B": 0}
    assert "eggs" not in result


def test_eggs_accumulate_without_opening_record():
    from domino_video.rules import ScoringRules

    result = ScoringRules().settle(
        "withEggs", "fish", None, {"A": 14, "B": 14}, {"A": 0, "B": 0}, {"A": 8, "B": 12}, 48
    )
    assert result == {
        "reason": "eggs",
        "winner_team": None,
        "round_score": {"A": 0, "B": 0},
        "total_score": {"A": 0, "B": 0},
        "fence": {"A": 8, "B": 12},
        "eggs": 76,
    }


def test_without_eggs_keeps_both_fences_and_independent_thresholds():
    from domino_video.rules import ScoringRules

    result = ScoringRules().settle(
        "withoutEggs", "fish", None, {"A": 12, "B": 14}, {"A": 0, "B": 0}, {"A": 8, "B": 12}, 0
    )
    assert result["total_score"] == {"A": 0, "B": 26}
    assert result["fence"] == {"A": 20}
    assert result["winner_team"] is None


@pytest.mark.parametrize("limit", [50, 101])
def test_simultaneous_limit_and_draw(limit):
    from domino_video.rules import ScoringRules

    rules = ScoringRules()
    assert rules.game_result({"A": limit, "B": limit}, limit) == {"draw": True}
    assert rules.game_result({"A": limit, "B": limit + 10}, limit) == {"winner": "A"}
    assert rules.game_result({"B": limit + 10, "A": limit}, limit) == {"winner": "A"}


@pytest.mark.parametrize(
    "hand",
    [
        ["0-0", "1-1", "2-2", "3-3", "4-4", "0-1", "0-2"],
        ["0-0", "0-1", "0-2", "0-3", "0-4", "0-5", "1-1"],
    ],
)
def test_rejects_deal_requiring_redeal(example, hand):
    from domino_video.validator import GameValidator, InvalidGame

    rest = sorted({f"{a}-{b}" for a in range(7) for b in range(a, 7)} - set(hand))
    players = list(example["rounds"][0]["deal"])
    example["rounds"][0]["deal"] = {
        players[0]: hand,
        **{players[i + 1]: rest[i * 7 : (i + 1) * 7] for i in range(3)},
    }
    with pytest.raises(InvalidGame, match="пересдач"):
        GameValidator().validate(example)


def test_allows_reversed_start_orientation(example):
    from domino_video.validator import GameValidator

    moves = example["rounds"][2]["moves"]
    moves[0]["stone"] = "6-4"
    for move in moves[1:]:
        if move["action"] in ("left", "right"):
            move["action"] = "right" if move["action"] == "left" else "left"
    assert GameValidator().validate(example) == example
