def test_preserves_manual_game_events_from_observations():
    from domino_video.reconstruct import GameReconstructor
    from domino_video.vision import Observation, StoneObservation

    # A single unambiguous start must survive duplicate frames.
    stone = StoneObservation((1, 1), (800, 280, 60, 120))
    rows = [Observation(t, [stone], [[], [], [], []], 0, False, True) for t in [1, 1.25, 1.5, 1.75]]
    result = GameReconstructor().extract(rows)
    assert result[0]["events"][0]["stone"] == "1-1"
    assert len(result[0]["events"]) == 1
    assert result[0]["complete"] is False


def test_matches_every_manually_transcribed_move(observations, sample_game):
    from domino_video.reconstruct import GameReconstructor

    r = GameReconstructor()
    actual = r.build(
        r.extract(observations), next(o.names for o in observations if o.names), "withoutEggs", 50
    )
    assert actual == sample_game


def test_rejects_truncated_ending(observations):
    import pytest

    from domino_video.reconstruct import GameReconstructor, ReconstructionError

    r = GameReconstructor()
    with pytest.raises(ReconstructionError):
        r.build(
            r.extract([o for o in observations if o.time < 270]),
            next(o.names for o in observations if o.names),
            "withoutEggs",
            50,
        )


def test_rejects_hidden_remaining_tiles(observations):
    import pytest

    from domino_video.reconstruct import GameReconstructor, ReconstructionError

    r = GameReconstructor()
    rounds = r.extract(observations)
    rounds[2]["remaining"][2] = []
    with pytest.raises(ReconstructionError):
        r.build(rounds, next(o.names for o in observations if o.names), "withoutEggs", 50)


def test_rejects_truncated_beginning(observations):
    import pytest

    from domino_video.reconstruct import GameReconstructor, ReconstructionError

    r = GameReconstructor()
    with pytest.raises(ReconstructionError):
        r.build(
            r.extract([o for o in observations if o.time > 30]),
            next(o.names for o in observations if o.names),
            "withoutEggs",
            50,
        )


def test_delayed_start_keeps_chronological_sides():
    """14:31 recording, 18.4–23.9s: 1-1 starts, then 6-1 on its left."""
    import gzip
    import json

    from domino_video.reconstruct import GameReconstructor
    from domino_video.vision import Observation, StoneObservation

    with gzip.open("tests/fixtures/delayed_start.json.gz", "rt", encoding="utf8") as f:
        rows = json.load(f)
    for row in rows:
        for field in ("board", "uncertain_board"):
            row[field] = [StoneObservation(**s) for s in row[field]]
    events = GameReconstructor().extract([Observation(**row) for row in rows])[0]["events"]
    assert [(e["stone"], e["action"]) for e in events[:3]] == [
        ("1-1", "start"),
        ("1-6", "left"),
        ("4-6", "left"),
    ]


def test_side_is_not_inferred_from_unrelated_later_rearrangement():
    from domino_video.reconstruct import GameReconstructor
    from domino_video.vision import Observation, StoneObservation

    first = StoneObservation((1, 1), (800, 280, 60, 120))
    second = StoneObservation((1, 2), (900, 300, 120, 60))
    rows = [Observation(0, [], [[], [], [], []], 0, False, True)]
    rows += [Observation(t, [first], [[], [], [], []], 0, False, True) for t in (0.5, 1, 1.5)]
    rows += [Observation(t, [second], [[], [], [], []], 1, False, True) for t in (2, 2.5, 3)]
    rows.append(Observation(4, [first, second], [[], [], [], []], 2, False, True))
    events = GameReconstructor().extract(rows)[0]["events"]
    assert events[1]["stone"] == "1-2"
    assert events[1]["action"] is None
