from copy import deepcopy

from domino_video.corrections import Corrections
from domino_video.manager import ParseManager
from domino_video.reconstruct import GameReconstructor
from domino_video.video import VideoTimeline


def with_independent_terminal_hands(observations):
    """Synthetic before/after evidence layered over the recorded legal journal."""
    rows = deepcopy(observations)
    rounds = GameReconstructor().extract(rows)
    deck = {f"{a}-{b}" for a in range(7) for b in range(a, 7)}
    for rnd in rounds:
        known = {e["stone"] for e in rnd["events"]}
        known.update(stone for hand in rnd["remaining"] for stone in hand)
        missing = deck - known
        if len(missing) != 1 or not any(not hand for hand in rnd["remaining"]):
            continue
        seat = next(i for i, hand in enumerate(rnd["remaining"]) if not hand)
        previous = next(o for o in reversed(rows) if not o.reveal and o.time < rnd["end"])
        for delta in (0.2, 0.1):
            before = deepcopy(previous)
            before.time = rnd["end"] - delta
            before.hands[seat] = list(missing)
            rows.append(before)
        for obs in rows:
            if rnd["end"] <= obs.time < rnd["end"] + 3 and obs.reveal:
                obs.reveal_valid = (True, True, True, True)
    return sorted(rows, key=lambda o: o.time)


def test_unique_recording_does_not_decode_recovery_windows(
    tmp_path, observations, sample_game, monkeypatch
):
    observations = with_independent_terminal_hands(observations)

    class Reader:
        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            yield from ((obs.time, obs) for obs in observations)

        def interval_frames(self, path, windows):
            raise AssertionError("Unique logical game must not request dense recovery")

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    monkeypatch.setattr(
        "domino_video.recognition_samples.RecognitionSamples.write_stones", lambda *args: False
    )
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"fixture")
    assert (
        ParseManager(Reader(), Recognizer()).run(
            [source], tmp_path / "out", "withoutEggs", 50, Corrections()
        )
        == 0
    )


def test_logical_resolution_preserves_originals_on_failure(observations):
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(observations)
    rounds[0]["events"].pop(2)
    rounds[0]["events"].pop(2)
    original = deepcopy(rounds)
    # Missing observations cannot be repaired by selecting the first legal tile.
    assert not reconstructor.resolve_logically(rounds, observations, "withoutEggs", 50)
    assert rounds == original


def test_logical_resolution_fills_unknown_confirmed_slot(observations, sample_game):
    observations = with_independent_terminal_hands(observations)
    reconstructor = GameReconstructor()
    rounds = reconstructor.extract(observations)
    assert reconstructor.resolve_logically(rounds, observations, "withoutEggs", 50)
    missing = rounds[0]["events"][5]
    missing["stone"] = None
    assert reconstructor.resolve_logically(rounds, observations, "withoutEggs", 50)
    names = [
        player["name"]
        for player in sorted(
            [player for team in sample_game["teams"] for player in team["players"]],
            key=lambda p: p["seat"],
        )
    ]
    assert reconstructor.build(rounds, names, "withoutEggs", 50) == sample_game
    assert rounds[0]["events"][5]["stone"] is not None
    assert not reconstructor.recovery_windows(rounds)


def test_complete_accounting_does_not_hide_an_independent_unknown_gap(observations):
    import pytest

    from domino_video.reconstruct import ReconstructionError

    reconstructor = GameReconstructor()
    observations = with_independent_terminal_hands(observations)
    rounds = reconstructor.extract(observations)
    assert reconstructor.resolve_logically(rounds, observations, "withoutEggs", 50)
    rounds[0]["unresolved"].append(
        dict(
            stone=None,
            interval=[39, 41],
            candidates=[],
            metadata={"unknown_count": None},
        )
    )
    with pytest.raises(ReconstructionError, match="независимые"):
        reconstructor.build(rounds, ["A", "B", "C", "D"], "withoutEggs", 50)
