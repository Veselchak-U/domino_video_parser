from domino_video.stone_recovery import StoneRecovery


def test_unfinished_round_reads_known_gap_to_available_source_end():
    rnd = dict(
        start=10,
        end=None,
        complete=False,
        events=[],
        observation_times=[10, 12, 16],
        unresolved=[dict(interval=[13, 15])],
    )
    assert StoneRecovery().windows([rnd]) == [(12, 16)]


def test_unfinished_round_without_known_gap_does_not_get_final_window():
    rnd = dict(
        start=10,
        end=None,
        complete=False,
        events=[],
        remaining=[[], [], [], []],
        observation_times=[10, 16],
        unresolved=[],
    )
    assert StoneRecovery().windows([rnd]) == []


def test_windows_merge_only_inside_round_and_include_long_hidden_final_slot():
    rounds = [
        dict(
            start=0,
            end=10,
            complete=True,
            events=[dict(time=2, confirmed_at=3, stone="1-1")],
            remaining=[[], [], [], []],
            observation_times=[0, 3, 9, 10],
            unresolved=[],
        ),
        dict(
            start=10,
            end=20,
            complete=True,
            events=[],
            observation_times=[10, 20],
            unresolved=[dict(interval=[11, 19])],
        ),
    ]
    assert StoneRecovery().windows(rounds) == [(3, 10), (10, 20)]
