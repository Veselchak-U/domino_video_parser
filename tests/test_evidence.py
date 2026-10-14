import numpy as np
import pytest

from domino_video.evidence import SourceEvidence
from domino_video.vision import Observation, StoneObservation


def observation(time=1, **kwargs):
    return Observation(time, [], [[], [], [], []], None, False, True, **kwargs)


def test_lru_byte_budget_preserves_results_and_isolates_sources():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    first = SourceEvidence("first", image_budget_bytes=36)
    second = SourceEvidence("second", image_budget_bytes=36)
    first.merge(observation(), fields=("counts",))
    for timestamp in (1, 2):
        first.cache_image(timestamp, image)
    first.image_at(1)
    first.cache_image(3, image)
    assert first.image_at(2) is None
    assert first.image_bytes == 36
    assert first.has_analysis(1)
    assert first.missing_fields(1, ("counts", "scores")) == {"scores"}
    assert not first.missing_fields(1, ("counts",), mode="motion")
    assert not second.observations and second.image_at(1) is None
    first.cache_image(4, np.zeros((100, 100), dtype=np.uint8))
    assert first.image_at(4) is None and first.image_bytes == 36
    first.close()
    assert first.image_bytes == 0
    with pytest.raises(RuntimeError, match="закрыт"):
        first.image_at(1)
    second.close()


def test_merge_completes_partial_fields_preserves_confirmations_and_attempts():
    with SourceEvidence("source") as evidence:
        original = observation(counts=(7, None, 6, None), scores=(3, 4))
        original.board = [StoneObservation((None, 3), (1, 2, 3, 4))]
        original.ocr_attempts = [{"field": "counts", "raw": "?"}]
        evidence.merge(original)
        additional = observation(counts=(7, 5, 4, None), scores=None)
        additional.board = [StoneObservation((2, 3), (1, 2, 3, 4))]
        additional.ocr_attempts = [{"field": "counts", "raw": "5"}]
        evidence.merge(additional, mode="motion", fields=("counts",))
        evidence.merge(additional, mode="motion", fields=("counts",))
        result = evidence.observations[0]
        assert result.counts == (7, 5, 6, None) and result.scores == (3, 4)
        assert result.board[0].stone == "2-3"
        assert result.ocr_attempts == original.ocr_attempts + additional.ocr_attempts
        assert len(evidence.conflicts) == 1
        assert original.counts == (7, None, 6, None)
        assert len(evidence.observations) == 1


def test_error_closes_context_and_image_views_do_not_exceed_allocation_budget():
    evidence = SourceEvidence("source", image_budget_bytes=18)
    backing = np.zeros((100, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        with evidence:
            evidence.cache_image(1, backing[:2])
            assert evidence.image_at(1).base is None
            evidence.merge(observation(float("nan")))
    assert evidence.image_bytes == 0


def test_raw_image_budget_measurement():
    sizes = [
        np.empty((height, width, 3), dtype=np.uint8).nbytes
        for width, height in [(1608, 720), (1920, 1080)]
    ]
    assert sizes == [3473280, 6220800]
    assert [64 * 1024 * 1024 // size for size in sizes] == [19, 10]
