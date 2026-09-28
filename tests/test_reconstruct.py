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
