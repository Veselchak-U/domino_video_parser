import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from domino_video.recognition_plan import recognition_requests
from domino_video.remaining_evidence import confirm_empty_remaining
from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation


def terminal_case():
    deck = [f"{a}-{b}" for a in range(7) for b in range(a, 7)]
    rnd = dict(
        start=0,
        end=20,
        complete=True,
        remaining=[["0-0"], ["0-1"], ["0-2"], []],
        remaining_confirmed=[True] * 4,
        events=[
            dict(stone=s, time=i / 2)
            for i, s in enumerate(deck)
            if s not in {"0-0", "0-1", "0-2", "6-6"}
        ],
    )
    rows = []
    for time in (21, 22, 23, 24, 25):
        row = Observation(time, [], deepcopy(rnd["remaining"]), None, True, True)
        row.reveal_valid = (True,) * 4
        row.reveal_panels_complete = (True,) * 4
        rows.append(row)
    return rnd, rows


def test_confirmed_empty_missing_terminal_requests_bounded_points_and_collects_proof():
    rnd, rows = terminal_case()
    requests = recognition_requests([rnd], rows)
    assert [t for t, f in requests.items() if f.reveal_points] == [22, 23, 24, 25]


def test_confirmed_empty_collects_independent_zero_proof_idempotently():
    rnd, rows = terminal_case()
    for row in rows:
        row.reveal_points = (0, 1, 2, 0)
    confirm_empty_remaining([rnd], rows)
    assert rnd["remaining_evidence"][0]["times"] == [21, 22, 23, 24, 25]
    assert StoneRecovery._final_empty_evidence(
        rnd, rnd["events"], ["6-6"], rnd["remaining_evidence"]
    )
    confirm_empty_remaining([rnd], rows)
    assert len(rnd["remaining_evidence"]) == 1
    assert not recognition_requests([rnd], rows)


def test_confirmed_empty_requests_require_distinct_panel_times():
    rnd, rows = terminal_case()
    assert not recognition_requests([rnd], [rows[0], deepcopy(rows[0])])
    rows = [rows[0], rows[1], deepcopy(rows[1]), deepcopy(rows[1]), deepcopy(rows[1])]
    assert list(recognition_requests([rnd], rows)) == [21, 22]


@pytest.mark.parametrize("kind", ["full", "duplicate", "two_empty", "incomplete", "unconfirmed"])
def test_no_extra_points_for_non_terminal_confirmed_empty(kind):
    rnd, rows = terminal_case()
    if kind == "full":
        rnd["events"].append(dict(stone="6-6", time=19))
    elif kind == "duplicate":
        rnd["events"][-1]["stone"] = rnd["events"][0]["stone"]
    elif kind == "two_empty":
        rnd["remaining"][2] = []
    elif kind == "incomplete":
        rnd["complete"] = False
    else:
        rnd["remaining_confirmed"][0] = False
    assert not recognition_requests([rnd], rows)


@pytest.mark.parametrize("kind", ["positive", "hand", "adjacent", "duplicate_pts", "partial"])
def test_confirmed_empty_proof_rejects_conflicts_and_boundaries(kind):
    rnd, rows = terminal_case()
    rows = rows[:2]
    for row in rows:
        row.reveal_points = (0, 1, 2, 0)
    rounds = [rnd]
    if kind == "positive":
        rows[-1].reveal_points = (0, 1, 2, 1)
    elif kind == "hand":
        rows[-1].hands[3] = ["6-6"]
    elif kind == "adjacent":
        rounds.append(dict(start=22, complete=False))
    elif kind == "duplicate_pts":
        rows[-1].time = rows[0].time
    else:
        rows[-1].reveal_panels_complete = (True, True, True, False)
    confirm_empty_remaining(rounds, rows)
    assert not rnd.get("remaining_evidence")


def test_cached_real_finished_panels_collect_strict_terminal_evidence():
    fixture = Path("tests/fixtures/confirmed_terminal_points_probe.json.gz")
    with gzip.open(fixture, "rt", encoding="utf-8") as stream:
        probe = json.load(stream)
    assert (
        probe["source_sha256"] == "e4c95c8dd07c7b110f8aa4ffcd3872be18c87ad8123bb13e7b6f306b5ff56f89"
    )
    assert probe["models"]
    for item in probe["results"]:
        rnd = item["raw_round"]
        assert not rnd["unresolved"]
        rows = [Observation(**raw) for raw in item["observations"]]
        visual_only = deepcopy(rows)
        for row in visual_only:
            row.reveal_points = None
            row.ocr_attempts = []
        requests = recognition_requests([rnd], visual_only)
        assert [t for t, fields in requests.items() if fields.reveal_points] == item[
            "requested_pts"
        ]
        confirm_empty_remaining([rnd], rows)
        assert rnd["remaining_evidence"] == [item["proof"]]
        assert (
            StoneRecovery._final_empty_evidence(
                rnd, rnd["events"], item["missing"], rnd["remaining_evidence"]
            )
            == item["strict_final_empty_evidence"]
        )
        assert not any(
            fields.reveal_points for fields in recognition_requests([rnd], rows).values()
        )
