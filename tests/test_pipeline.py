import pytest

from domino_video.cli import main
from domino_video.pipeline import ObservationPipeline, WorkerSettings


@pytest.mark.parametrize("count", [1, 4, 24])
def test_workers_follow_current_machine(monkeypatch, count):
    monkeypatch.setattr("domino_video.pipeline.os.process_cpu_count", lambda: count, raising=False)
    assert WorkerSettings().default() == count


def test_windows_worker_limit_and_unknown_cpu(monkeypatch):
    monkeypatch.setattr("domino_video.pipeline.sys.platform", "win32")
    monkeypatch.setattr("domino_video.pipeline.os.process_cpu_count", lambda: 128, raising=False)
    assert WorkerSettings().default() == 61
    monkeypatch.setattr("domino_video.pipeline.os.process_cpu_count", lambda: None)
    assert WorkerSettings().default() == 1


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "62"])
def test_invalid_workers_rejected_before_video(value, monkeypatch):
    monkeypatch.setattr("domino_video.pipeline.sys.platform", "win32")
    with pytest.raises(SystemExit) as error:
        main(["--workers", value, "--fish-variant", "withoutEggs", "--score-limit", "50"])
    assert error.value.code == 2


class Recognizer:
    def observe(self, image, timestamp, read_text):
        return timestamp, image, read_text


def test_sequential_preserves_ocr_schedule_and_closes_input():
    closed = []

    def frames():
        try:
            for t in (0, 1, 5, 6, 10):
                yield t, str(t)
        finally:
            closed.append(True)

    assert list(ObservationPipeline(1, recognizer=Recognizer()).observe(frames())) == [
        (0, "0", True),
        (1, "1", False),
        (5, "5", True),
        (6, "6", False),
        (10, "10", True),
    ]
    assert closed == [True]


@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
def test_parallel_bounds_queue_preserves_order_and_closes(monkeypatch, failure):
    consumed = []
    completed = []
    closed = []

    class Future:
        def __init__(self, args):
            self.args = args

        def result(self):
            if failure:
                raise failure("stop")
            completed.append(self.args[1])
            return self.args[1]

        def cancel(self):
            pass

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["mp_context"].get_start_method() == "spawn"

        def submit(self, fn, *args):
            return Future(args)

        def shutdown(self, **kwargs):
            closed.append(kwargs)

    monkeypatch.setattr("domino_video.pipeline.ProcessPoolExecutor", Pool)

    def frames():
        for t in range(30):
            consumed.append(t)
            assert len(consumed) - len(completed) <= 4
            yield t, None

    iterator = ObservationPipeline(2).observe(frames())
    if failure:
        with pytest.raises(failure):
            list(iterator)
    else:
        assert list(iterator) == list(range(30))
    assert closed == [dict(wait=True, cancel_futures=True)]


def test_real_spawn_pool_matches_sequential_and_leaves_no_children():
    import multiprocessing
    from pathlib import Path

    import cv2

    image = cv2.imread(str(Path(__file__).parent / "fixtures" / "frame_20.png"))

    def frames():
        for t in (0, 0.25, 0.5):
            yield t, image

    baseline = {p.pid for p in multiprocessing.active_children()}
    expected = list(ObservationPipeline(1).observe(frames()))
    actual = list(ObservationPipeline(2).observe(frames()))
    assert actual == expected
    assert {p.pid for p in multiprocessing.active_children()} == baseline


def test_real_spawn_pool_closes_on_consumer_interrupt():
    import multiprocessing
    from contextlib import closing

    import numpy as np

    baseline = {p.pid for p in multiprocessing.active_children()}
    frames = ((t, np.zeros((720, 1608, 3), dtype=np.uint8)) for t in range(12))
    with pytest.raises(KeyboardInterrupt):
        with closing(ObservationPipeline(2).observe(frames)) as results:
            next(results)
            raise KeyboardInterrupt
    assert {p.pid for p in multiprocessing.active_children()} == baseline


def crash_worker(*args):
    import os

    os._exit(7)


def test_real_crashed_worker_releases_pool_and_next_source_succeeds(monkeypatch):
    import multiprocessing
    from concurrent.futures.process import BrokenProcessPool

    import numpy as np

    import domino_video.pipeline as pipeline

    baseline = {p.pid for p in multiprocessing.active_children()}
    original = pipeline._observe_frame
    monkeypatch.setattr(pipeline, "_observe_frame", crash_worker)
    frames = ((t, None) for t in range(10))
    with pytest.raises(BrokenProcessPool):
        list(ObservationPipeline(2).observe(frames))
    assert {p.pid for p in multiprocessing.active_children()} == baseline
    monkeypatch.setattr(pipeline, "_observe_frame", original)
    frames = ((0, np.zeros((720, 1608, 3), dtype=np.uint8)) for _ in range(1))
    assert len(list(ObservationPipeline(2).observe(frames))) == 1
    assert {p.pid for p in multiprocessing.active_children()} == baseline
