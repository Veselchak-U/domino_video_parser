import gzip
import json
from copy import deepcopy

import pytest

from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation, StoneObservation


def evidence(index):
    with gzip.open("tests/fixtures/placement_sides1444.json.gz", "rt", encoding="utf8") as stream:
        case = json.load(stream)["cases"][index]
    rows = []
    for item in case["native"]:
        for key in ("board", "uncertain_board"):
            item[key] = [StoneObservation(tuple(t["values"]), tuple(t["box"])) for t in item[key]]
        rows.append(Observation(**item))
    return case["round"], rows


@pytest.mark.parametrize("index,side", [(0, "left"), (1, "right")])
def test_real_single_visible_or_geometrically_known_end_resolves_side(index, side):
    rnd, rows = evidence(index)
    rec = GameReconstructor()
    rec.integrate_recovery([rnd], rows)
    options = rec._round_options(rnd, rnd["events"], list("ABCD"), index + 1)
    assert len(options) == 1
    assert options[0]["moves"][-1]["action"] == side
    old = deepcopy(rnd["events"])
    rec.integrate_recovery([rnd], rows)
    assert rnd["events"] == old


@pytest.mark.parametrize("kind", ["one_frame", "one_projection", "overlap", "missing_source"])
def test_hidden_end_needs_repeated_unique_physical_evidence(kind):
    rnd, rows = evidence(0)
    if kind == "one_frame":
        rows = [o for o in rows if o.time < 92.90 or o.time > 92.96]
    elif kind == "one_projection":
        for o in rows:
            if o.time > 92.94:
                o.board = [t for t in o.board if t.stone != "1-2"]
    elif kind == "missing_source":
        for o in rows:
            if o.time < 92.897:
                o.board = [t for t in o.board if t.stone != "4-4"]
    else:
        for o in rows:
            if o.time > 92.90:
                o.uncertain_board = [
                    StoneObservation(t.values, (t.box[0] + 25, *t.box[1:]))
                    for t in o.uncertain_board
                ]
    rec = GameReconstructor()
    rec.integrate_recovery([rnd], rows)
    assert not rnd["events"][-1].get("placement_links")
    assert len(rec._round_options(rnd, rnd["events"], list("ABCD"), 1)) == 2


@pytest.mark.parametrize("kind", ["two_ends", "interior_neighbour"])
def test_contact_link_does_not_choose_between_ends_or_use_interior(kind):
    rnd, _ = evidence(1)
    event = rnd["events"][-1]
    event["placement_links"] = [
        dict(stone=key) for key in (["0-4", "3-3"] if kind == "two_ends" else ["2-3"])
    ]
    assert len(GameReconstructor()._round_options(rnd, rnd["events"], list("ABCD"), 5)) == 2


@pytest.mark.parametrize("kind", ["two_frames", "overlap"])
def test_direct_end_contact_requires_settlement_without_overlap(kind):
    rnd, rows = evidence(1)
    if kind == "two_frames":
        rows = [o for o in rows if o.time < 475.585]
    else:
        for o in rows:
            o.board = [
                StoneObservation(t.values, (t.box[0] + 40, *t.box[1:])) if t.stone == "0-3" else t
                for t in o.board
            ]
    rec = GameReconstructor()
    rec.integrate_recovery([rnd], rows)
    assert not rnd["events"][-1].get("placement_links")
    assert len(rec._round_options(rnd, rnd["events"], list("ABCD"), 5)) == 2


def test_two_old_values_projecting_to_hidden_end_remain_ambiguous():
    rnd, rows = evidence(0)
    for o in rows:
        if o.time >= 92.897:
            continue
        old = next((t for t in o.board if t.stone == "4-4"), None)
        if old:
            o.board = [t for t in o.board if t.stone != "0-0"]
            o.board.append(StoneObservation((0, 0), old.box))
    rec = GameReconstructor()
    rec.integrate_recovery([rnd], rows)
    assert not rnd["events"][-1].get("placement_links")
    assert len(rec._round_options(rnd, rnd["events"], list("ABCD"), 1)) == 2


@pytest.mark.parametrize("kind", ["duplicate_pts", "exchanged_continuations"])
def test_hidden_contact_rejects_competing_contours_and_track_exchange(kind):
    rnd, rows = evidence(0)
    for obs in rows:
        if obs.time < 92.91:
            continue
        candidates = [t for t in obs.uncertain_board if t.box[1] > 450]
        for tile in candidates:
            if kind == "duplicate_pts":
                shift = 2
            else:
                shift = 3 if obs.time < 92.94 else -3
            obs.uncertain_board.append(
                StoneObservation(tile.values, (tile.box[0], tile.box[1] + shift, *tile.box[2:]))
            )
    rec = GameReconstructor()
    rec.integrate_recovery([rnd], rows)
    assert not rnd["events"][-1].get("placement_links")
    assert len(rec._round_options(rnd, rnd["events"], list("ABCD"), 1)) == 2
