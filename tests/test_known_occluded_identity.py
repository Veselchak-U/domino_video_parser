import pytest

from domino_video.stone_recovery import StoneRecovery
from domino_video.vision import Observation, StoneObservation


@pytest.mark.parametrize("kind", ["settled", "cancelled", "competing"])
def test_existing_read_stone_can_reconcile_anchored_duplicate_but_not_other_slot(kind):
    anchors = [
        StoneObservation((0, 6), (300, 160, 50, 100)),
        StoneObservation((0, 4), (400, 250, 100, 50)),
    ]
    rows = []
    for time, x, values in [
        (0, 700, (6, 6)),
        (0.1, 550, (6, 6)),
        (0.2, 400, (6, 6)),
        (0.5, 300, (None, None)),
        (0.6, 300, (None, None)),
        (0.7, 300, (None, None)),
    ]:
        if kind == "cancelled" and time <= 0.2:
            x = 1100 - x
        tile = StoneObservation(values, (x, 250, 100, 50))
        rows.append(
            Observation(
                time,
                anchors + ([tile] if values[0] is not None else []),
                [[], [], [], []],
                0,
                False,
                True,
                uncertain_board=[tile] if values[0] is None else [],
            )
        )
        if kind == "competing" and time >= 0.5:
            rows[-1].uncertain_board.append(StoneObservation((None, None), (310, 250, 100, 50)))
    duplicate = dict(
        time=0.5,
        interval=[0.4, 0.5],
        stone=None,
        reason="occluded",
        region=[300, 250, 100, 50],
        candidates=[],
        anchors=[dict(stone=tile.stone, center=list(tile.center), size=100) for tile in anchors],
    )
    independent = dict(duplicate, region=[900, 250, 100, 50])
    independent_metadata = dict(duplicate, metadata=dict(unknown_count=None))
    rnd = dict(
        events=[dict(time=0, stone="6-6")],
        unresolved=[duplicate, independent, independent_metadata],
    )
    StoneRecovery()._discard_known_occluded_slots(rnd, rows)
    if kind == "settled":
        assert rnd["unresolved"] == [independent, independent_metadata]
        assert rnd["ignored_fragments"][0]["stone"] == "6-6"
    else:
        assert rnd["unresolved"] == [duplicate, independent, independent_metadata]
