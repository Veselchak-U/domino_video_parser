from copy import deepcopy

import pytest

from domino_video.corrections import Corrections


def test_partial_game_can_supply_unknown_stone_without_exact_time():
    document = {
        "format": "kozel-game-v2",
        "status": "needs_review",
        "source": {"sha256": "a" * 64, "game_number": 1},
        "fishVariant": "withEggs",
        "scoreLimit": 101,
        "teams": [
            {
                "id": "A",
                "name": "Team A",
                "players": [{"name": "a", "seat": 1}, {"name": "c", "seat": 3}],
            },
            {
                "id": "B",
                "name": "Team B",
                "players": [{"name": "b", "seat": 2}, {"name": "d", "seat": 4}],
            },
        ],
        "rounds": [
            {
                "number": 1,
                "deal": {name: None for name in "abcd"},
                "moves": [
                    {
                        "event_id": "x",
                        "recognition_status": "supplied",
                        "player": "a",
                        "action": "start",
                        "stone": "1-1",
                        "interval": [1, 2],
                        "reason": "unknown",
                        "candidates": [],
                    }
                ],
                "origin_events": [
                    {
                        "event_id": "x",
                        "recognition_status": "unresolved",
                        "player": "a",
                        "action": "start",
                        "stone": None,
                        "interval": [1, 2],
                        "reason": "unknown",
                        "candidates": [],
                    }
                ],
                "result": None,
                "evidence": {},
            }
        ],
        "result": None,
    }
    raw = [
        {
            "events": [
                {
                    "event_id": "x",
                    "seat": 0,
                    "action": "start",
                    "stone": None,
                    "time": 1,
                    "interval": [1, 2],
                }
            ],
            "remaining": [[], [], [], []],
        }
    ]
    corrected, teams = Corrections(deepcopy(document)).apply("a" * 64, 1, raw)
    assert corrected[0]["events"][0]["stone"] == "1-1"
    assert corrected[0]["events"][0]["interval"] == [1, 2]
    assert corrected[0]["events"][0]["time_is_anchor"] is True
    assert teams == document["teams"]


def test_builder_retains_unknown_hand_and_stable_identity():
    from domino_video.partial_game import build_partial_game, validate_partial_game

    raw = [
        {
            "events": [
                {"seat": 1, "action": "right", "stone": None, "time": 3, "interval": [2, 4]}
            ],
            "initial_hand": ["0-0"],
            "score": [2, 3],
        }
    ]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    validate_partial_game(doc)
    assert doc["rounds"][0]["deal"] == dict.fromkeys("abcd")
    move = doc["rounds"][0]["moves"][0]
    assert move["stone"] is None
    assert move["player"] == "b"
    assert doc["result"] is None
    assert doc["rounds"][0]["result"] is None
    assert move == doc["rounds"][0]["origin_events"][0]
    with pytest.raises(ValueError):
        doc["status"] = "validated"
        validate_partial_game(doc)


def partial_fixture():
    from domino_video.partial_game import build_partial_game

    raw = [
        {
            "events": [
                {
                    "seat": None,
                    "action": None,
                    "stone": None,
                    "time": 2,
                    "interval": [1, 3],
                    "metadata": {"unknown_count": None, "order_candidates": []},
                }
            ],
            "remaining": [[], [], [], []],
        }
    ]
    return raw, build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)


def test_gap_accepts_sequence_and_keeps_origin():
    raw, doc = partial_fixture()
    origin = doc["rounds"][0]["moves"][0]
    first = dict(
        origin,
        event_id="first",
        parent_event_id=origin["event_id"],
        recognition_status="supplied",
        player="a",
        action="start",
        stone="1-1",
    )
    second = dict(first, event_id="second", player="b", action="right", stone="1-2")
    doc["rounds"][0]["moves"] = [first, second]
    result, _ = Corrections(doc).apply("a" * 64, 1, raw)
    assert [e["event_id"] for e in result[0]["events"]] == ["first", "second"]
    assert result[0]["origin_events"] == doc["rounds"][0]["origin_events"]
    assert result[0]["events"][1]["orientation"] == "1-2"


@pytest.mark.parametrize(
    "damage",
    ["unmarked", "unknown_id", "parent", "interval", "duplicate", "unknown_field", "status"],
)
def test_partial_rejects_bad_edit(damage):
    from domino_video.partial_game import validate_partial_game

    _, doc = partial_fixture()
    move = doc["rounds"][0]["moves"][0]
    if damage == "unmarked":
        move["stone"] = "1-1"
    elif damage == "unknown_id":
        move["event_id"] = "other"
    elif damage == "parent":
        move.update(event_id="other", parent_event_id="foreign", recognition_status="supplied")
    elif damage == "interval":
        move["interval"] = [0, 4]
    elif damage == "duplicate":
        doc["rounds"][0]["moves"].append(deepcopy(move))
    elif damage == "unknown_field":
        move["unexpected"] = True
    else:
        doc["status"] = "validated"
    with pytest.raises(ValueError):
        validate_partial_game(doc)


def test_partial_fill_retains_unknown_and_hash_binding():
    raw, doc = partial_fixture()
    doc["rounds"][0]["moves"][0].update(player="a", recognition_status="supplied")
    correction = Corrections(doc)
    result, _ = correction.apply("a" * 64, 1, raw)
    assert result[0]["events"][0]["stone"] is None
    assert result[0]["events"][0]["seat"] == 0
    with pytest.raises(ValueError, match="Хеш"):
        correction.check_sources({"b" * 64})
    correction.check_settings("withEggs", 101)
    with pytest.raises(ValueError, match="Вариант"):
        correction.check_settings("withoutEggs", 101)
    with pytest.raises(ValueError, match="лимит"):
        correction.check_settings("withEggs", 50)


def test_supplied_pass_preserves_anchors_without_becoming_placement():
    raw, doc = partial_fixture()
    origin = doc["rounds"][0]["moves"][0]
    placement = dict(
        origin,
        event_id="placed",
        parent_event_id=origin["event_id"],
        recognition_status="supplied",
        player="a",
        action="start",
        stone="1-1",
    )
    passed = dict(placement, event_id="passed", player="b", action="pass")
    passed.pop("stone")
    doc["rounds"][0]["moves"] = [placement, passed]
    result, _ = Corrections(doc).apply("a" * 64, 1, raw)
    assert len(result[0]["events"]) == 1
    assert result[0]["supplied_passes"][0]["before_event_id"] == "placed"
    assert result[0]["supplied_passes"][0]["after_event_id"] is None


def test_builder_id_does_not_depend_on_array_index():
    from domino_video.partial_game import build_partial_game

    raw, doc = partial_fixture()
    expected = doc["rounds"][0]["moves"][0]["event_id"]
    raw[0]["events"].insert(0, {"seat": 0, "action": "start", "stone": "1-1", "time": 0})
    result = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    assert result["rounds"][0]["moves"][1]["event_id"] == expected


def test_missing_last_event_exports_gap_and_manual_fill_reconstructs(observations, sample_game):
    from domino_video.partial_game import build_partial_game
    from domino_video.reconstruct import GameReconstructor

    reconstructor = GameReconstructor()
    raw = reconstructor.extract(observations)
    names = [
        player["name"]
        for player in sorted(
            [p for t in sample_game["teams"] for p in t["players"]], key=lambda p: p["seat"]
        )
    ]
    for rnd, expected_round in zip(raw, sample_game["rounds"], strict=True):
        rnd["unresolved"] = []
        rnd["events"] = [
            dict(
                seat=names.index(move["player"]),
                action=move["action"],
                stone="-".join(sorted(move["stone"].split("-"))),
                orientation=move["stone"],
                time=rnd["start"] + index,
            )
            for index, move in enumerate(
                [m for m in expected_round["moves"] if m["action"] != "pass"]
            )
        ]
    removed = raw[-1]["events"].pop()
    original = deepcopy(raw)
    doc = build_partial_game(raw, names, "withoutEggs", 50, "a" * 64, 1)
    assert raw == original
    gap = doc["rounds"][-1]["moves"][-1]
    assert gap["action"] == "unknown"
    assert gap["stone"] is None
    assert gap["metadata"]["unknown_count"] is None
    expected = [m for m in sample_game["rounds"][-1]["moves"] if m["action"] != "pass"][-1]
    gap.update(
        recognition_status="supplied",
        player=expected["player"],
        action=expected["action"],
        stone=expected["stone"],
    )
    corrected, _ = Corrections(doc).apply("a" * 64, 1, raw)
    assert corrected[-1]["events"][-1]["stone"] == removed["stone"]
    assert reconstructor.build(corrected, names, "withoutEggs", 50) == sample_game
    bad_orientation = deepcopy(corrected)
    event = bad_orientation[-1]["events"][-1]
    event["orientation"] = "-".join(reversed(event["orientation"].split("-")))
    with pytest.raises(ValueError):
        reconstructor.build(bad_orientation, names, "withoutEggs", 50)


def test_unresolved_visual_slot_exports_and_partial_fill_survives():
    from domino_video.partial_game import build_partial_game

    raw = [
        {
            "events": [],
            "unresolved": [
                {
                    "seat": 1,
                    "action": None,
                    "stone": None,
                    "candidates": ["2-3"],
                    "time": 2,
                    "interval": [1, 3],
                    "anchors": [{"stone": "1-2", "center": [1, 2]}],
                }
            ],
        }
    ]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    move = doc["rounds"][0]["moves"][0]
    assert move["player"] == "b"
    assert move["candidates"] == [{"stone": "2-3"}]
    move.update(recognition_status="supplied", action="right")
    corrected, _ = Corrections(doc).apply("a" * 64, 1, raw)
    assert len(corrected[0]["events"]) == 1
    assert corrected[0]["events"][0]["stone"] is None
    assert corrected[0]["unresolved"] == []
    exported = build_partial_game(corrected, list("abcd"), "withEggs", 101, "a" * 64, 1)
    assert exported["rounds"][0]["moves"][0]["stone"] is None
    assert exported["rounds"][0]["origin_events"][0]["action"] == "unknown"


def test_resolved_visual_slot_is_not_duplicated_in_partial_document():
    from domino_video.partial_game import build_partial_game

    raw = [
        {
            "events": [{"seat": 0, "action": "start", "stone": "1-1", "time": 2}],
            "unresolved": [
                {
                    "seat": 0,
                    "action": None,
                    "stone": None,
                    "candidates": ["1-1"],
                    "time": 2,
                    "interval": [1, 3],
                }
            ],
        }
    ]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    assert len(doc["rounds"][0]["moves"]) == 1


def test_v2_batch_corrections_leave_other_sources_and_games_untouched():
    raw, doc = partial_fixture()
    corrections = Corrections(doc)
    corrections.check_sources({"a" * 64, "b" * 64})
    corrections.check_games("a" * 64, 2)
    for sha, game_number in (("b" * 64, 1), ("a" * 64, 2)):
        result, teams = corrections.apply(sha, game_number, raw)
        assert result == raw
        assert result is not raw and result[0] is not raw[0]
        assert teams is None


def test_score_conflict_does_not_invent_unknown_move():
    from domino_video.partial_game import build_partial_game

    raw = [{"events": [{"seat": 0, "action": "start", "stone": "1-1", "time": 1}], "score": [3, 4]}]
    diagnostic = {
        "code": "score_conflict",
        "round": 1,
        "details": {"expected": [5, 6], "observed": [3, 4]},
    }
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1, [diagnostic])
    assert len(doc["rounds"][0]["moves"]) == 1
    assert doc["rounds"][0]["moves"][0]["recognition_status"] == "confirmed"
    assert doc["rounds"][0]["evidence"]["score"] == [3, 4]
    assert doc["rounds"][0]["result"] is None
    assert doc["diagnostics"] == [diagnostic]


@pytest.mark.parametrize("field,value", [("interval", [0, 3]), ("seat", 1), ("stone", "2-2")])
def test_partial_import_rejects_origin_changed_since_export(field, value):
    raw, doc = partial_fixture()
    raw[0]["events"][0][field] = value
    with pytest.raises(ValueError, match="Исходная ссылка"):
        Corrections(doc).apply("a" * 64, 1, raw)


def test_no_move_review_retains_origin_and_other_unknown_on_reexport():
    from domino_video.partial_game import build_partial_game

    raw, _ = partial_fixture()
    raw[0]["events"].append(dict(seat=None, action=None, stone=None, time=4, interval=[4, 5]))
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    rejected = doc["rounds"][0]["moves"][0]
    rejected.update(recognition_status="supplied", review_resolution="no_move")
    corrected, _ = Corrections(doc).apply("a" * 64, 1, raw)
    assert len(corrected[0]["events"]) == 1
    assert corrected[0]["events"][0]["time"] == 4
    assert corrected[0]["events"][0]["stone"] is None
    exported = build_partial_game(corrected, list("abcd"), "withEggs", 101, "a" * 64, 1)
    assert exported["rounds"][0]["moves"][0]["review_resolution"] == "no_move"
    assert exported["rounds"][0]["moves"][1]["recognition_status"] == "unresolved"
    assert exported["rounds"][0]["origin_events"] == doc["rounds"][0]["origin_events"]


def test_no_move_cannot_discard_confirmed_origin():
    from domino_video.partial_game import build_partial_game

    raw = [{"events": [dict(seat=0, action="start", stone="1-1", time=1)]}]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    doc["rounds"][0]["moves"][0].update(
        recognition_status="supplied",
        review_resolution="no_move",
        player=None,
        action="unknown",
        stone=None,
    )
    with pytest.raises(ValueError, match="unresolved"):
        Corrections(doc)


@pytest.mark.parametrize("tamper", ["confirmed_status", "unresolved_metadata"])
def test_origin_classification_cannot_be_tampered_to_authorize_discard_or_split(tamper):
    from domino_video.partial_game import build_partial_game

    raw = [{"events": [dict(seat=0, action="start", stone="1-1", time=1)]}]
    if tamper == "unresolved_metadata":
        raw[0]["events"][0].update(seat=None, action=None, stone=None)
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    origin = doc["rounds"][0]["origin_events"][0]
    move = doc["rounds"][0]["moves"][0]
    if tamper == "confirmed_status":
        origin["recognition_status"] = "unresolved"
        move.update(
            recognition_status="supplied",
            review_resolution="no_move",
            player=None,
            action="unknown",
            stone=None,
        )
    else:
        origin["metadata"] = dict(unknown_count=None, order_candidates=[])
        move.update(
            event_id="new",
            parent_event_id=origin["event_id"],
            recognition_status="supplied",
            player="a",
            action="start",
            stone="1-1",
        )
    with pytest.raises(ValueError, match="Исходная ссылка"):
        Corrections(doc).apply("a" * 64, 1, raw)


def test_no_move_derived_failure_gap_is_not_duplicated_when_other_round_is_unknown():
    from domino_video.partial_game import build_partial_game

    raw = [
        dict(start=0, end=3, events=[dict(seat=0, action="start", stone="1-1", time=1)]),
        dict(
            start=4,
            end=5,
            events=[dict(seat=None, action=None, stone=None, time=4, interval=[4, 5])],
        ),
    ]
    diagnostics = [dict(round=1, details=dict(first_failure=dict(interval=[2, 3], event=1)))]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1, diagnostics)
    doc["rounds"][0]["moves"][-1].update(recognition_status="supplied", review_resolution="no_move")
    corrected, _ = Corrections(doc).apply("a" * 64, 1, raw)
    exported = build_partial_game(
        corrected, list("abcd"), "withEggs", 101, "a" * 64, 1, diagnostics
    )
    assert len(exported["rounds"][0]["moves"]) == 2
    assert exported["rounds"][0]["moves"][-1]["review_resolution"] == "no_move"
    assert exported["rounds"][1]["moves"][0]["stone"] is None
    assert exported["diagnostics"] == diagnostics


def test_no_move_cannot_suppress_new_missing_stones_marker():
    from domino_video.partial_game import build_partial_game

    raw = [dict(start=0, end=3, events=[dict(seat=0, action="start", stone="1-1", time=1)])]
    diagnostics = [dict(round=1, details=dict(first_failure=dict(interval=[2, 3], event=1)))]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1, diagnostics)
    doc["rounds"][0]["moves"][-1].update(recognition_status="supplied", review_resolution="no_move")
    corrected, _ = Corrections(doc).apply("a" * 64, 1, raw)
    corrected[0]["missing"] = ["4-6"]
    exported = build_partial_game(
        corrected, list("abcd"), "withEggs", 101, "a" * 64, 1, diagnostics
    )
    unresolved = [
        move
        for move in exported["rounds"][0]["moves"]
        if move["recognition_status"] == "unresolved"
    ]
    assert len(unresolved) == 1
    assert unresolved[0]["candidates"] == [{"stone": "4-6"}]


def test_early_unknown_slots_do_not_hide_missing_final_interval():
    from domino_video.partial_game import build_partial_game

    deck = {f"{a}-{b}" for a in range(7) for b in range(a, 7)}
    raw = [
        dict(
            complete=True,
            start=0,
            end=10,
            events=[dict(seat=0, action="start", stone="1-1", time=9)],
            unresolved=[
                dict(
                    seat=None,
                    action=None,
                    stone=None,
                    time=2,
                    interval=[2, 3],
                    event_id=f"early-{i}",
                )
                for i in range(3)
            ],
            missing=["4-6"],
            remaining=[sorted(deck - {"1-1", "4-6"}), [], [], []],
            remaining_confirmed=[True] * 4,
        )
    ]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    moves = doc["rounds"][0]["moves"]
    assert [move["event_id"] for move in moves[:3]] == [f"early-{i}" for i in range(3)]
    gap = moves[-1]
    assert gap["interval"] == [9, 10]
    assert gap["recognition_status"] == "unresolved"
    assert gap["stone"] is None
    assert gap["candidates"] == [{"stone": "4-6"}]
    assert gap["metadata"]["unknown_count"] is None


def test_imported_interval_anchor_does_not_reorder_source_origins_on_reexport():
    from domino_video.partial_game import build_partial_game

    raw = [
        dict(
            events=[
                dict(seat=0, action="start", stone="1-1", time=2),
                dict(seat=None, action=None, stone=None, time=3, interval=[1, 3]),
            ]
        )
    ]
    doc = build_partial_game(raw, list("abcd"), "withEggs", 101, "a" * 64, 1)
    corrected, _ = Corrections(doc).apply("a" * 64, 1, raw)
    assert corrected[0]["events"][1]["time"] == 1
    assert corrected[0]["events"][1]["time_is_anchor"] is True
    exported = build_partial_game(corrected, list("abcd"), "withEggs", 101, "a" * 64, 1)
    assert exported["rounds"][0]["moves"] == doc["rounds"][0]["moves"]
    assert exported["rounds"][0]["origin_events"] == doc["rounds"][0]["origin_events"]
