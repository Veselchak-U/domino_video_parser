from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import pytest
from test_recognition_diagnostics import ControlledOCRAdapter, ControlledScreenRecognizer

from domino_video import pipeline
from domino_video.gpu_worker import GPUWorker
from domino_video.ocr import DeviceOCR
from domino_video.reconstruct import GameReconstructor
from domino_video.vision import Observation, PreparedObservation, StoneObservation


class RecoveryScreenRecognizer(ControlledScreenRecognizer):
    def prepare(self, image, time, read_text=False):
        return PreparedObservation(
            deepcopy(image),
            dict(
                names=[np.zeros((10, 20, 3), np.uint8)] * 4,
                counts=[np.zeros((10, 20, 3), np.uint8)] * 3,
            ),
        )


def initialize_recovery_worker():
    pipeline._initialize_worker()
    pipeline._recognizer = RecoveryScreenRecognizer()


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
@pytest.mark.parametrize("workers", [1, 2])
def test_recovered_moves_and_proofs_match_across_process_modes(monkeypatch, mode, workers):
    def executor(**kwargs):
        kwargs["initializer"] = initialize_recovery_worker
        return ProcessPoolExecutor(**kwargs)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", executor)
    if mode == "gpu":
        monkeypatch.setattr(
            pipeline,
            "DeviceOCR",
            lambda mode, setting: DeviceOCR(
                mode,
                "1",
                available=lambda: True,
                worker_factory=lambda: GPUWorker(ControlledOCRAdapter),
            ),
        )
    first = StoneObservation((1, 1), (800, 280, 50, 100))
    rows = [
        Observation(t, [first], [hand, [], [], []], 0, False, True)
        for t, hand in [
            (1, ["1-6", "6-6"]),
            (1.25, ["1-6", "6-6"]),
            (1.5, ["1-6", "6-6"]),
            (2, ["6-6"]),
            (2.25, ["6-6"]),
        ]
    ]
    unknown = StoneObservation((None, None), (870, 280, 50, 100))
    rows += [
        Observation(t, [first], [["6-6"], [], [], []], 1, False, True, uncertain_board=[unknown])
        for t in [3, 3.25]
    ]
    shifted = StoneObservation((1, 1), (700, 280, 50, 100))
    late = StoneObservation((1, 2), (770, 280, 50, 100))
    short = StoneObservation((2, 6), (840, 280, 50, 100))
    rows += [
        Observation(t, [shifted, late], [["6-6"], [], [], []], 1, False, True)
        for t in [4, 4.25, 4.5]
    ]
    rows += [
        Observation(5, [shifted, late, short], [["6-6"], [], [], []], 2, False, True),
        Observation(5.25, [], [[], [], [], []], None, True, True),
    ]

    def frames():
        for row in rows:
            yield row.time, row

    actual = list(
        pipeline.ObservationPipeline(workers, RecoveryScreenRecognizer(), device=mode).observe(
            frames()
        )
    )
    r = GameReconstructor()
    actual_rounds = r.extract(actual)
    expected_rounds = r.extract(rows)
    dense = [
        Observation(t, [shifted, late, short], [[], [], [], []], 2, False, True, dense=True)
        for t in [5, 5.05, 5.1]
    ]
    r.integrate_recovery(actual_rounds, dense)
    r.integrate_recovery(expected_rounds, dense)
    rnd = actual_rounds[0]
    expected = expected_rounds[0]
    assert rnd["events"] == expected["events"]
    assert rnd["stone_recovery"] == expected["stone_recovery"]
    assert {e["method"] for e in rnd["stone_recovery"]} == {
        "hand_difference",
        "late_reading",
        "animation",
    }
