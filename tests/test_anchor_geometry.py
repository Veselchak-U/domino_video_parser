import gzip
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_anchor_recovery import native_rows

from domino_video.anchor_geometry import anchor_matches, repeated_identity, static_slot_bridge
from domino_video.vision import Observation, StoneObservation


def evidence():
    rows = native_rows()
    early = rows[0]
    hidden = next(t for t in early.uncertain_board if t.box[0] == 1192)
    anchors = [next(t for t in early.board if t.stone == key) for key in ("1-4", "0-1")]
    return hidden, anchors, early.board, [row for row in rows if row.time > 125]


def test_real_rotated_identity_has_repeated_independent_geometry():
    hidden, anchors, board, late = evidence()
    proof = repeated_identity(hidden, anchors, board, late)
    assert proof["stone"] == "4-5"
    assert len({row["time"] for row in proof["readings"]}) >= 3
    assert max(row["error"] for row in proof["readings"]) < 1.3
    assert all(0.72 < row["scale"] < 0.75 for row in proof["readings"])


@pytest.mark.parametrize(
    "fault",
    [
        "one_reading",
        "same_pts",
        "one_anchor",
        "duplicate_early",
        "duplicate_late",
        "two_values",
        "degenerate",
        "dimensions",
    ],
)
def test_real_identity_rejects_insufficient_or_conflicting_geometry(fault):
    hidden, anchors, board, late = evidence()
    if fault == "one_reading":
        late = late[:1]
    elif fault == "same_pts":
        late = late[:1] * 3
    elif fault == "one_anchor":
        anchors = anchors[:1]
    elif fault == "duplicate_early":
        board.append(deepcopy(anchors[0]))
    elif fault == "duplicate_late":
        for row in late:
            row.board.append(deepcopy(next(t for t in row.board if t.stone == "1-4")))
    elif fault == "two_values":
        for row in late:
            tile = deepcopy(next(t for t in row.board if t.stone == "4-5"))
            row.board.append(replace(tile, values=(5, 5)))
    elif fault == "degenerate":
        anchors[1] = replace(anchors[1], box=anchors[0].box)
    elif fault == "dimensions":
        for row in late:
            tile = next(t for t in row.board if t.stone == "0-1")
            x, y, w, h = tile.box
            row.board[row.board.index(tile)] = replace(
                tile, box=(x - w / 2, y - h / 2, w * 2, h * 2)
            )
    assert repeated_identity(hidden, anchors, board, late) is None


def tile(value, x, y):
    return StoneObservation(values=value, box=(x - 10, y - 20, 20, 40))


def test_pair_collision_is_ambiguous_even_with_repeated_frames():
    early = [tile((1, 1), 100, 100), tile((2, 2), 160, 100), tile((3, 3), 100, 160)]
    hidden = tile((None, None), 130, 130)
    late = [
        tile((1, 1), 300, 300),
        tile((2, 2), 360, 300),
        tile((3, 3), 300, 240),
        tile((5, 5), 330, 330),
        tile((6, 6), 270, 270),
    ]
    assert {p["stone"] for p in anchor_matches(hidden, early, early, late)} == {"5-5", "6-6"}
    rows = [
        Observation(time=i, board=late, hands={}, active=None, reveal=False, supported=True)
        for i in range(3)
    ]
    assert repeated_identity(hidden, early, early, rows) is None


def test_reflection_is_not_a_direct_similarity():
    early = [tile((1, 1), 100, 100), tile((2, 2), 160, 100)]
    hidden = tile((None, None), 130, 130)
    late = [tile((1, 1), 300, 300), tile((2, 2), 360, 300), tile((5, 5), 330, 270)]
    assert anchor_matches(hidden, early, early, late) == []


def test_single_rotating_pip_alias_is_not_a_repeated_late_identity():
    hidden, anchors, board, late = evidence()
    alias = deepcopy(late[0])
    alias.time -= 0.02
    candidate = next(tile for tile in alias.board if tile.stone == "4-5")
    alias.board[alias.board.index(candidate)] = replace(candidate, values=(4, 4))
    proof = repeated_identity(hidden, anchors, board, [alias, *late])
    assert proof["stone"] == "4-5"
    assert proof["unconfirmed_values"] == {"4-4": [alias.time]}


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "unobserved_gap",
        "missing_local_anchor",
        "two_contours",
        "known_competitor",
        "different_place",
    ],
)
def test_real_occlusion_bridge_requires_independent_stationary_geometry(fault):
    with gzip.open(
        "tests/fixtures/anchor_recovery/full_first_round.json.gz", "rt", encoding="utf8"
    ) as stream:
        data = json.load(stream)
    rows = []
    for row in data["observations"]:
        if not 114.336 <= row["time"] <= 114.57:
            continue
        for key in ("board", "uncertain_board"):
            row[key] = [StoneObservation(**tile) for tile in row[key]]
        rows.append(Observation(**row))
    first = next(t for t in rows[0].uncertain_board if 1180 < t.box[0] < 1210)
    last = next(t for t in rows[-1].uncertain_board if 1180 < t.box[0] < 1210)
    anchors = [t for t in rows[0].board if t.stone in {"1-4", "0-1"}]
    if fault == "unobserved_gap":
        rows = [rows[0], rows[-1]]
    elif fault == "missing_local_anchor":
        for row in rows:
            row.board = [t for t in row.board if t.stone != "0-1"]
    elif fault == "two_contours":
        rows[0].uncertain_board.append(first)
    elif fault == "known_competitor":
        rows[1].board.append(replace(first, values=(5, 5)))
    elif fault == "different_place":
        x, y, w, h = last.box
        last = replace(last, box=(x + 100, y, w, h))
    proof = static_slot_bridge(first, last, anchors, rows)
    if fault is None:
        assert {"0-1", "0-3"} <= set(proof["anchors"])
        assert len(proof["times"]) > 10
    else:
        assert proof is None
