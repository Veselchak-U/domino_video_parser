import gzip
import json
from pathlib import Path

import cv2
import pytest

from domino_video.indicator_evidence import preserve_player_candidates
from domino_video.recognition_plan import recognition_requests
from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation, ScreenRecognizer, StoneObservation

FIXTURES = Path("tests/fixtures/active_method")
CASES = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf8"))


def observations(name):
    with gzip.open(FIXTURES / f"{name}.json.gz", "rt", encoding="utf8") as source:
        rows = json.load(source)
    result = []
    for row in rows:
        method = row.pop("active_method", None)
        for field in ("board", "uncertain_board"):
            row[field] = [StoneObservation(**s) for s in row[field]]
        obs = Observation(**row)
        obs.active_method = method
        result.append(obs)
    return result


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_real_digits_override_wrong_decorative_color(case):
    obs = (
        ScreenRecognizer()
        .prepare(cv2.imread(str(FIXTURES / f"{case['name']}.png")), case["time"])
        .observation
    )
    assert obs.active_method == "timer"
    assert obs.active == case["actual_seat"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_real_weak_history_does_not_exclude_counter_confirmed_player(case):
    rows = observations(case["name"])
    rounds = GameReconstructor().extract(rows)
    event = next(e for e in rounds[0]["events"] if e["stone"] == case["stone"])
    assert rounds[0].get("indicator_unreliable")
    assert event["seats"] == [0, 1, 2, 3]
    assert case["actual_seat"] in event["seats"]
    requests = recognition_requests(rounds, rows)
    index = rounds[0]["events"].index(event)
    until = rounds[0]["events"][index + 1]["time"]
    count_times = [
        t for t, fields in requests.items() if fields.counts and event["time"] < t < until
    ]
    assert len(count_times) == 2


@pytest.mark.parametrize("method", [None, "arc", "none", "ambiguous_arc"])
def test_missing_or_nonweak_method_does_not_mark_round(method):
    rows = observations("19-44")
    for obs in rows:
        obs.active_method = method
    assert not GameReconstructor().extract(rows)[0].get("indicator_unreliable")


def test_native_only_weak_lookback_is_applied_before_recovery(monkeypatch):
    from domino_video.stone_recovery import StoneRecovery

    rows = observations("19-44")
    weak = next(o for o in rows if o.active_method == "avatar_color")
    weak.active_method = None
    rounds = GameReconstructor().extract(rows)
    weak.active_method = "avatar_color"
    assert weak.time < rounds[0]["start"]
    original = StoneRecovery._earlier_placements

    def verify(self, rnd, native):
        assert rnd["indicator_unreliable"]
        assert rnd["events"][0]["seats"] == [0, 1, 2, 3]
        return original(self, rnd, native)

    monkeypatch.setattr(StoneRecovery, "_earlier_placements", verify)
    GameReconstructor().integrate_recovery(rounds, [weak])
    snapshot = json.dumps(rounds, sort_keys=True)
    GameReconstructor().integrate_recovery(rounds, [weak])
    assert json.dumps(rounds, sort_keys=True) == snapshot


@pytest.mark.parametrize("method", ["animation", "later_frame", "late_reading"])
def test_recovered_value_does_not_prove_its_indicator_player(method):
    event = dict(stone="1-1", seat=2, seats=[2], recovery=dict(method=method, seat=3))
    hand = dict(stone="1-2", seat=0, seats=[0], recovery=dict(method="hand_difference", seat=1))
    rnd = dict(
        indicator_unreliable=True, events=[event, hand], unresolved=[dict(seat=3, seats=[3])]
    )
    preserve_player_candidates(rnd)
    assert event["seats"] == [0, 1, 2, 3]
    assert event["recovery"]["seat"] is None
    assert hand["seats"] == [0]
    assert rnd["unresolved"][0]["seats"] == [0, 1, 2, 3]


def test_prior_reveal_separates_the_next_round_source_history():
    from domino_video.vision import StoneObservation

    stone = StoneObservation((1, 1), (800, 280, 60, 120))
    rows = [Observation(1, [], [[], [], [], []], 2, False, True, active_method="avatar_color")]
    rows.append(Observation(1.5, [], [[], [], [], []], None, True, True))
    rows.extend(
        Observation(t, [stone], [[], [], [], []], 3, False, True, active_method="arc")
        for t in [2, 2.5, 3]
    )
    assert not GameReconstructor().extract(rows)[0].get("indicator_unreliable")
