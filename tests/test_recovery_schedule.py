from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from domino_video.evidence import SourceEvidence
from domino_video.manager import ParseManager
from domino_video.ocr_result import OCRResult
from domino_video.recognition_plan import OCRFields
from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation, ScreenRecognizer


def row(timestamp):
    return Observation(timestamp, [], [[], [], [], []], None, False, True)


def test_fallback_windows_include_only_residual_rounds():
    resolved = dict(
        start=0,
        end=5,
        complete=True,
        events=[],
        unresolved=[],
        logically_resolved=True,
    )
    residual = dict(
        start=5,
        end=10,
        complete=True,
        events=[],
        observation_times=[5, 10],
        unresolved=[dict(interval=[6, 9])],
    )
    assert GameReconstructor().recovery_windows([resolved, residual]) == [(5, 10)]


def test_motion_frames_reuse_pts_in_order_and_skip_completed_modes():
    calls = []
    image = np.zeros((3, 3, 3), dtype=np.uint8)

    class Reader:
        def sampled_interval_frames(self, path, windows, sample_fps):
            calls.append((windows, sample_fps))
            for timestamp in (0.1, 0.2, 0.3):
                yield timestamp, image

    manager = ParseManager(reader=Reader())
    with SourceEvidence("source") as evidence:
        manager._evidence = evidence
        for timestamp in (0.15, 0.2):
            evidence.merge(row(timestamp), fields=("counts",))
            evidence.cache_image(timestamp, image)
        evidence.merge(row(0.2), mode="motion")
        frames = list(manager._recovery_frames("source", [(0, 0.4)]))
        assert [timestamp for timestamp, _ in frames] == [0.1, 0.15, 0.3]
        assert calls == [([(0, 0.4)], 10)]
        assert not evidence.missing_fields(0.15, ("counts",), mode="motion")
        assert len(evidence.observations) == 2


def test_read_fields_reuses_existing_ocr_attempts_without_second_read(monkeypatch):
    image = cv2.imread("tests/fixtures/frame_20.png")
    assert image is not None
    manager = ParseManager(reader=SimpleNamespace())
    manager._score_searched = True
    progress = SimpleNamespace(message=lambda *args: None)
    monkeypatch.setattr(
        "domino_video.manager.recognition_requests", lambda *args: {20: OCRFields(counts=True)}
    )
    monkeypatch.setattr("domino_video.manager.start_requests", lambda *args: {})
    monkeypatch.setattr("domino_video.manager.confirm_empty_remaining", lambda *args: None)
    monkeypatch.setattr("domino_video.manager.confirm_round_starts", lambda *args: None)
    calls = []

    class Pipeline:
        def __init__(self, *args, **kwargs):
            self.fields = kwargs["fields"]

        def observe(self, frames):
            for timestamp, pixels in frames:
                prepared = ScreenRecognizer().prepare(pixels, timestamp, self.fields[timestamp])

                def read(crop):
                    calls.append(crop.shape)
                    return OCRResult.from_rows([(None, "6", 0.99)])

                yield prepared.finish(read)

    monkeypatch.setattr("domino_video.manager.ObservationPipeline", Pipeline)
    with SourceEvidence("source") as evidence:
        manager._evidence = evidence
        evidence.cache_image(20, image)
        original = row(20)
        original.scores = (3, 4)
        original.ocr_attempts = [dict(field="score", accepted_text="3/50")]
        observations = [evidence.merge(original)]
        for _ in range(2):
            manager._read_fields("source", [], observations, 1, progress, "cpu", "auto")
        assert len(calls) == 3
        assert observations[0].counts == (6, 6, 6, 6)
        assert observations[0].scores == (3, 4)
        assert len(observations[0].ocr_attempts) == 4


def test_incomplete_recovery_preserves_unfinished_end(monkeypatch):
    seen = []

    def integrate(self, rounds, observations):
        seen.append(rounds[0]["end"])
        rounds[0]["new_evidence"] = True

    monkeypatch.setattr("domino_video.stone_recovery.StoneRecovery.integrate", integrate)
    rnd = dict(start=1, end=None, complete=False, observation_times=[1, 4, 7], events=[])
    GameReconstructor().integrate_recovery([rnd], [])
    assert seen == [7]
    assert rnd["end"] is None and not rnd["complete"]
    assert rnd["new_evidence"]


def test_failed_partial_decode_releases_source_and_context():
    closed = []

    class Reader:
        def sampled_interval_frames(self, path, windows, sample_fps):
            try:
                yield 0.1, np.zeros((2, 2, 3), dtype=np.uint8)
                raise ValueError("failed partial decode")
            finally:
                closed.append(True)

    manager = ParseManager(reader=Reader())
    evidence = SourceEvidence("source")
    with pytest.raises(ValueError, match="failed partial decode"):
        with evidence:
            manager._evidence = evidence
            list(manager._recovery_frames("source", [(0, 1)]))
    assert closed == [True] and evidence.image_bytes == 0
