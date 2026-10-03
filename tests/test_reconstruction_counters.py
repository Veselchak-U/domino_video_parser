import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from domino_video.reconstruction_counters import count_readings


def recording():
    return json.loads(Path("tests/fixtures/ahead_counter.json").read_text(encoding="utf8"))


def test_real_selection_counters_do_not_reject_preceding_placement():
    fixture = recording()
    before = deepcopy(fixture)
    readings = count_readings(fixture["counters"], fixture["played"], fixture["next_event"])
    assert readings == [Counter({3: 2}), Counter({2: 2}), Counter({3: 2}), Counter({4: 2})]
    assert fixture == before


@pytest.mark.parametrize(
    "counts,next_event",
    [
        ([3, 2, 2, 4], None),
        ([3, 2, 2, 4], {"seats": [1, 3]}),
        ([3, 2, 1, 4], {"seats": [2]}),
        ([3, 1, 3, 4], {"seats": [2]}),
        ([3, 1, 2, 5], {"seats": [2]}),
        ([None, 2, 2, 4], {"seats": [2]}),
        ([3, 2, 2, 4], {"seats": []}),
    ],
)
def test_unexplained_counter_conflicts_remain(counts, next_event):
    fixture = recording()
    counters = [dict(time=1, counts=counts), dict(time=2, counts=counts)]
    assert count_readings(counters, fixture["played"], next_event) == [
        Counter() if value is None else Counter({value: 2}) for value in counts
    ]


def test_correct_and_conflicting_evidence_remain_separate():
    fixture = recording()
    counters = fixture["counters"] + [dict(time=70, counts=[3, 2, 1, 4])]
    readings = count_readings(counters, fixture["played"], fixture["next_event"])
    assert readings[2] == Counter({3: 2, 1: 1})


def test_exact_current_counts_need_no_next_event():
    fixture = recording()
    counts = [3, 2, 3, 4]
    assert count_readings([dict(time=1, counts=counts)], fixture["played"], None) == [
        Counter({value: 1}) for value in counts
    ]


@pytest.mark.parametrize("seat", range(4))
def test_single_selection_can_belong_to_any_next_player(seat):
    fixture = recording()
    counts = [3, 2, 3, 4]
    selected = list(counts)
    selected[seat] -= 1
    assert count_readings(
        [dict(time=1, counts=selected)], fixture["played"], {"seats": [seat]}
    ) == [Counter({value: 1}) for value in counts]


@pytest.mark.parametrize("actual_seat", [1, 2])
@pytest.mark.parametrize("rotation", [0, 3])
def test_explained_selection_binds_the_actual_next_player(actual_seat, rotation):
    from domino_video.reconstruct import GameReconstructor

    deck = [f"{a}-{b}" for a in range(7) for b in range(a, 7) if f"{a}-{b}" not in ("1-1", "1-2")]
    left = [s for s in deck if "1" not in s.split("-")][: (6 if actual_seat == 1 else 7)]
    rest = [s for s in deck if s not in left]
    upper_count = 7 if actual_seat == 1 else 6
    remaining = [rest[:6], left, rest[6 : 6 + upper_count], rest[6 + upper_count :]]
    remaining = remaining[-rotation:] + remaining[:-rotation] if rotation else remaining
    counts = [6, 6, 7, 7]
    counts = counts[-rotation:] + counts[:-rotation] if rotation else counts
    events = [
        dict(time=1, stone="1-1", seat=rotation, seats=[rotation], action="start"),
        dict(
            time=3,
            stone="1-2",
            seat=None,
            seats=[(1 + rotation) % 4, (2 + rotation) % 4],
            action="right",
        ),
    ]
    raw = dict(
        remaining=remaining,
        end=5,
        indicator_unreliable=True,
        counters=[dict(time=t, counts=counts) for t in [1.5, 1.6]],
    )
    trace = []
    options = GameReconstructor()._round_options(raw, events, list("ABCD"), 1, trace)
    if actual_seat == 1:
        assert len(options) == 1
        assert options[0]["moves"][-1]["player"] == "ABCD"[(1 + rotation) % 4]
    else:
        assert options == []
        assert trace[1]["rejections"]["hand_counter_conflict"]["count"] == 1
