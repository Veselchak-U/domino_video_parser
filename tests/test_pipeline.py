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


def test_motion_pass_never_reads_text_or_initializes_gpu(monkeypatch):
    from types import SimpleNamespace

    calls = []

    class MotionRecognizer:
        def prepare(self, image, timestamp, read_text, read_motion=False):
            calls.append((timestamp, read_text, read_motion))
            return SimpleNamespace(observation=SimpleNamespace(time=timestamp, dense=False))

        def close(self):
            calls.append("closed")

    def forbidden(*args, **kwargs):
        pytest.fail("Motion pass initialized OCR")

    monkeypatch.setattr("domino_video.pipeline.DeviceOCR", forbidden)
    frames = ((t, None) for t in (0, 0.1, 0.2))
    rows = list(ObservationPipeline(1, recognizer=MotionRecognizer(), motion=True).observe(frames))
    assert [(o.time, o.dense) for o in rows] == [(0, True), (0.1, True), (0.2, True)]
    assert calls == [(0, False, True), (0.1, False, True), (0.2, False, True), "closed"]


def test_real_motion_pool_matches_sequential_and_closes():
    import multiprocessing

    import cv2

    image = cv2.imread("tests/fixtures/frame_20.png")
    baseline = {p.pid for p in multiprocessing.active_children()}
    expected = list(ObservationPipeline(1, motion=True).observe((t, image) for t in (0, 0.1)))
    actual = list(ObservationPipeline(2, motion=True).observe((t, image) for t in (0, 0.1)))
    assert actual == expected
    assert all(o.dense and not o.ocr_attempts for o in actual)
    assert {p.pid for p in multiprocessing.active_children()} == baseline


@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("motion", [False, True])
def test_parallel_bounds_queue_preserves_order_and_closes(monkeypatch, failure, motion):
    consumed = []
    completed = []
    closed = []

    from concurrent.futures import Future as RealFuture

    class Future(RealFuture):
        def __init__(self, args):
            super().__init__()
            self.args = args
            if failure:
                self.set_exception(failure("stop"))
            else:
                self.set_result(args[1])

        def result(self):
            result = super().result()
            completed.append(self.args[1])
            return result

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

    iterator = ObservationPipeline(2, motion=motion).observe(frames())
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


@pytest.mark.parametrize("motion", [False, True])
def test_real_spawn_pool_closes_on_consumer_interrupt(motion):
    import multiprocessing
    from contextlib import closing

    import numpy as np

    baseline = {p.pid for p in multiprocessing.active_children()}
    frames = ((t, np.zeros((720, 1608, 3), dtype=np.uint8)) for t in range(12))
    with pytest.raises(KeyboardInterrupt):
        with closing(ObservationPipeline(2, motion=motion).observe(frames)) as results:
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


@pytest.mark.parametrize("failure", [None, ValueError])
def test_refills_behind_slow_head_without_exceeding_window(monkeypatch, failure):
    from concurrent.futures import Future
    from concurrent.futures import wait as real_wait

    import domino_video.pipeline as pipeline

    submitted, closed = [], []
    first = None
    emitted = []

    class GuardedFuture(Future):
        def result(self, *args, **kwargs):
            assert self.done(), "Waited for the slow head instead of refilling"
            return super().result(*args, **kwargs)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, image, timestamp, read_text):
            nonlocal first
            future = GuardedFuture()
            submitted.append(timestamp)
            assert len(submitted) - len(emitted) <= 64
            if timestamp == 0:
                first = future
            elif failure and timestamp == 3:
                future.set_exception(failure("worker failed"))
            else:
                future.set_result((timestamp, read_text))
            return future

        def shutdown(self, **kwargs):
            closed.append(True)

    def controlled_wait(pending, **kwargs):
        assert len(pending) <= 4
        if not any(f.done() for f in pending):
            assert len(submitted) == 64
            first.set_result((0, True))
        return real_wait(pending, **kwargs)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(pipeline, "wait", controlled_wait, raising=False)
    frames = ((t, None) for t in range(200))

    def collect():
        for value in ObservationPipeline(2).observe(frames):
            emitted.append(value)

    if failure:
        with pytest.raises(ValueError, match="worker failed"):
            collect()
        assert not emitted
        assert first.cancelled()
    else:
        collect()
        assert emitted == [(t, t % 5 == 0) for t in range(200)]
    assert closed == [True]
