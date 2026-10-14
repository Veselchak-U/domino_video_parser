from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation


def test_weak_fragment_requires_unique_finished_game_and_never_assigns_value(observations):
    from copy import deepcopy

    import pytest
    from test_logical_recovery import with_independent_terminal_hands

    from domino_video.reconstruct import ReconstructionError

    rows = with_independent_terminal_hands(observations)
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(rows)
    assert reconstructor.resolve_logically(rounds, rows, "withoutEggs", 50)
    weak = dict(
        stone=None, candidates=[], reason="occluded", interval=[39, 41], region=[100, 100, 50, 100]
    )
    rounds[0]["unresolved"].append(weak)
    unfinished = deepcopy(rounds[:1])
    with pytest.raises(ReconstructionError):
        reconstructor.build(unfinished, list("ABCD"), "withoutEggs", 50, require_finished=False)
    assert unfinished[0]["unresolved"] == [weak]
    reconstructor.build(rounds, list("ABCD"), "withoutEggs", 50)
    assert weak not in rounds[0]["unresolved"]
    proof = rounds[0]["ignored_fragments"][-1]
    assert proof["method"] == "complete_journal_fragment"
    assert "stone" not in proof and "seat" not in proof
    assert len(set(proof["evidence"]["accounted"])) == 28
    assert proof["evidence"]["unique_journal"]


def test_zero_final_panel_proof_resolves_terminal_without_before_hand(observations):
    reconstructor = GameReconstructor()
    rnd = reconstructor.extract(observations)[0]
    seat = next(i for i, hand in enumerate(rnd["remaining"]) if not hand)
    rnd["remaining_confirmed"] = [True] * 4
    rnd["remaining_evidence"] = [
        dict(
            seat=seat + 1,
            method="zero_reveal_points",
            times=[rnd["end"] + 0.5, rnd["end"] + 1],
            points=[0, 0],
        )
    ]
    rnd["last_active"] = (seat + 1) % 4
    assert reconstructor.resolve_logically([rnd], [], "withoutEggs", 50)
    assert rnd["events"][-1]["seat"] == seat
    assert rnd["events"][-1]["recovery"]["evidence"]["remaining_evidence"]


def test_full_accounting_with_uncertain_player_or_side_still_requests_fallback():
    deck = [f"{a}-{b}" for a in range(7) for b in range(a, 7)]
    for event in [
        dict(time=5, interval=[4, 5], stone="1-6", seat=1, seats=[1, 2], action="right"),
        dict(time=5, interval=[4, 5], stone="1-6", seat=1, seats=[1], action=None),
    ]:
        rnd = dict(
            start=1,
            end=10,
            complete=True,
            events=[dict(time=1, stone="1-1", seat=0, seats=[0], action="start"), event],
            remaining=[[s for s in deck if s not in {"1-1", "1-6"}], [], [], []],
            observation_times=[1, 3, 6, 10],
            unresolved=[],
        )
        assert GameReconstructor().recovery_windows([rnd]) == [(1, 6)]


def test_confirmed_counter_conflict_rejects_single_player_branch():
    remaining = [f"{a}-{b}" for a in range(7) for b in range(a, 7) if (a, b) != (1, 1)]
    raw = dict(
        end=5,
        events=[dict(time=1, stone="1-1", seat=0, seats=[0], action="start")],
        remaining=[remaining[:6], remaining[6:13], remaining[13:20], remaining[20:]],
        counters=[dict(time=t, counts=(6, 0, 7, 7)) for t in [1.5, 2]],
    )
    assert GameReconstructor()._round_options(raw, raw["events"], ["A", "B", "C", "D"], 1) == []


def test_incomplete_recovery_does_not_receive_next_round_frames(monkeypatch):
    captured = []

    def integrate(self, rounds, observations):
        captured.append([o.time for o in observations])

    monkeypatch.setattr("domino_video.stone_recovery.StoneRecovery.integrate", integrate)
    rounds = [
        dict(start=1, end=None, complete=False, events=[], observation_times=[1, 4, 12]),
        dict(start=10, end=15, complete=True, events=[]),
    ]
    rows = [Observation(t, [], [[], [], [], []], None, False, True) for t in [1, 4, 10, 12]]
    GameReconstructor().integrate_recovery(rounds, rows)
    assert captured == [[1, 4], [10, 12]]
    assert rounds[0]["end"] is None
