import os
import time
from concurrent.futures import Future

import pytest


class EchoAdapter:
    def probe(self):
        return None

    def text(self, value):
        if value == "exit":
            os._exit(7)
        if value == "hang":
            time.sleep(60)
        return os.getpid(), value

    def close(self):
        pass


class HangingAdapter(EchoAdapter):
    def __init__(self):
        time.sleep(60)


class Prepared:
    crops = {"text": [1]}

    def __init__(self, value):
        self.value = value

    def finish(self, text):
        return text(self.value)


class FakeWorker:
    def __init__(self):
        self.closed = False
        self.probes = 0

    def probe(self):
        self.probes += 1

    def close(self):
        self.closed = True

    def submit(self, prepared):
        result = Future()
        result.set_result(prepared)
        return result


@pytest.mark.parametrize("mode", ["auto", "gpu"])
@pytest.mark.parametrize("setting", ["auto", "2"])
@pytest.mark.parametrize(
    "memory,count",
    [([2000, 1152, 256], 2), ([1999], 1), ([2000, 1151], 1), ([2000, 1152, 255], 1), ([None], 1)],
)
def test_memory_policy(mode, setting, memory, count):
    from domino_video.gpu_memory import MemorySnapshot
    from domino_video.ocr import DeviceOCR

    values = iter(memory)
    workers = []

    def factory():
        worker = FakeWorker()
        workers.append(worker)
        return worker

    with DeviceOCR(
        mode,
        setting,
        available=lambda: True,
        worker_factory=factory,
        memory_probe=lambda: MemorySnapshot(next(values), "unknown"),
    ) as gpu:
        assert gpu.enabled
        assert f"процессов: {count}" in gpu.description
        assert sum(not w.closed for w in workers) == count
    assert all(w.closed for w in workers)


def test_cpu_never_probes_gpu():
    from domino_video.ocr import DeviceOCR

    def unexpected():
        pytest.fail("GPU touched in CPU mode")

    with DeviceOCR(
        "cpu", "2", available=unexpected, memory_probe=unexpected, worker_factory=unexpected
    ) as gpu:
        assert not gpu.enabled


def test_process_ownership_and_native_exit():
    from domino_video.gpu_worker import GPUWorker

    first = GPUWorker(EchoAdapter)
    second = GPUWorker(EchoAdapter)
    try:
        a = first.submit(Prepared(1)).result(timeout=10)
        b = second.submit(Prepared(2)).result(timeout=10)
        assert len({a[0], b[0], os.getpid()}) == 3
        assert first.submit(Prepared(3)).result(timeout=10)[0] == a[0]
        failed = first.submit(Prepared("exit"))
        queued = first.submit(Prepared(5))
        for future in (failed, queued):
            with pytest.raises(RuntimeError):
                future.result(timeout=10)
        assert second.submit(Prepared(4)).result(timeout=10)[1] == 4
    finally:
        first.close()
        second.close()


def test_initialization_timeout_and_forced_shutdown():
    from domino_video.gpu_worker import GPUWorker

    with pytest.raises(TimeoutError):
        GPUWorker(HangingAdapter, startup_timeout=0.3, stop_timeout=0.1)
    worker = GPUWorker(EchoAdapter, stop_timeout=0.1)
    future = worker.submit(Prepared("hang"))
    pending = worker.submit(Prepared(1))
    start = time.monotonic()
    worker.close()
    assert time.monotonic() - start < 5
    assert future.done()
    assert pending.cancelled()


@pytest.mark.parametrize("value", ["0", "3", "-1", "1.5", "other"])
def test_invalid_gpu_workers(value):
    from domino_video.cli import main

    with pytest.raises(SystemExit) as error:
        main(["--gpu-workers", value, "--fish-variant", "withoutEggs", "--score-limit", "50"])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["auto", "gpu"])
@pytest.mark.parametrize("first_healthy", [True, False])
def test_second_start_failure_rechecks_first(mode, first_healthy):
    from domino_video.gpu_memory import MemorySnapshot
    from domino_video.ocr import DeviceOCR

    worker = FakeWorker()
    calls = []

    def probe():
        calls.append("probe")
        if not first_healthy:
            raise RuntimeError("driver failed")

    worker.probe = probe

    def factory():
        if calls:
            raise TimeoutError("second timeout")
        calls.append("start")
        return worker

    session = DeviceOCR(
        mode,
        available=lambda: True,
        worker_factory=factory,
        memory_probe=lambda: MemorySnapshot(4096),
    )
    if mode == "gpu" and not first_healthy:
        with pytest.raises(RuntimeError, match="driver failed"):
            with session:
                pass
    else:
        with session:
            assert session.enabled == first_healthy
            assert ("процессов: 1" in session.description) == first_healthy
    assert calls == ["start", "probe"]
    assert worker.closed


def test_distribution_cancel_interrupt_and_reselection():
    from domino_video.gpu_memory import MemorySnapshot
    from domino_video.ocr import DeviceOCR

    workers = []

    class PendingWorker(FakeWorker):
        def __init__(self):
            super().__init__()
            self.futures = []

        def submit(self, prepared):
            result = Future()
            self.futures.append(result)
            return result

        def close(self):
            super().close()
            for future in self.futures:
                future.cancel()

    def factory():
        worker = PendingWorker()
        workers.append(worker)
        return worker

    values = iter([2000, 1152, 256, None])
    args = dict(
        available=lambda: True,
        worker_factory=factory,
        memory_probe=lambda: MemorySnapshot(next(values)),
    )
    with pytest.raises(KeyboardInterrupt):
        with DeviceOCR("gpu", **args) as gpu:
            first = gpu.submit(1)
            second = gpu.submit(2)
            assert len(workers[0].futures) == len(workers[1].futures) == 1
            first.cancel()
            third = gpu.submit(3)
            assert len(workers[0].futures) == 2
            raise KeyboardInterrupt
    assert all(w.closed for w in workers)
    assert all(f.done() for f in (first, second, third))
    with DeviceOCR("gpu", **args) as gpu:
        assert "процессов: 1" in gpu.description


@pytest.mark.parametrize("setting", ["auto", "1", "2"])
def test_cli_passes_gpu_workers(tmp_path, monkeypatch, setting):
    from domino_video.cli import main

    (tmp_path / "sample.mp4").touch()
    calls = []
    monkeypatch.setattr(
        "domino_video.manager.ParseManager.run", lambda self, *args: calls.append(args) or 0
    )
    args = ["--input", str(tmp_path), "--fish-variant", "withoutEggs", "--score-limit", "50"]
    if setting != "auto":
        args += ["--gpu-workers", setting]
    assert main(args) == 0
    assert calls[0][-1] == setting


@pytest.mark.parametrize("cpu_workers", [1, 3])
def test_real_process_pipeline_order_bounds_and_failure_cleanup(monkeypatch, cpu_workers):
    from concurrent.futures import wait as real_wait

    from domino_video import pipeline
    from domino_video.gpu_memory import MemorySnapshot
    from domino_video.gpu_worker import GPUWorker
    from domino_video.ocr import DeviceOCR
    from domino_video.pipeline import ObservationPipeline

    class PreparedPool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, image, timestamp, read_text):
            result = Future()
            result.set_result(Prepared(timestamp if image is None else image))
            return result

        def shutdown(self, **kwargs):
            pass

    def bounded_wait(pending, **kwargs):
        assert len(pending) <= 2 * cpu_workers
        return real_wait(pending, **kwargs)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", PreparedPool)
    monkeypatch.setattr(pipeline, "wait", bounded_wait)

    workers = []

    def factory():
        worker = GPUWorker(EchoAdapter, stop_timeout=0.1)
        workers.append(worker)
        return worker

    monkeypatch.setattr(
        "domino_video.pipeline.DeviceOCR",
        lambda mode, setting: DeviceOCR(
            mode,
            setting,
            available=lambda: True,
            worker_factory=factory,
            memory_probe=lambda: MemorySnapshot(4096),
        ),
    )

    class Recognizer:
        def prepare(self, image, timestamp, read_text):
            return Prepared(timestamp if image is None else image)

    delivered = []

    def frames():
        for value in range(20):
            assert value - len(delivered) < 32 * cpu_workers
            yield value, None

    for pid, value in ObservationPipeline(cpu_workers, Recognizer(), device="gpu").observe(
        frames()
    ):
        assert pid != os.getpid()
        delivered.append(value)
    assert delivered == list(range(20))
    assert all(w.closed for w in workers)
    with pytest.raises(RuntimeError):
        list(
            ObservationPipeline(1, Recognizer(), device="gpu").observe(
                (frame for frame in [(0, "exit")])
            )
        )
    assert all(w.closed for w in workers)
    result = list(
        ObservationPipeline(1, Recognizer(), device="gpu").observe(
            (frame for frame in [(42, None)])
        )
    )
    assert result[0][1] == 42
    assert all(w.closed for w in workers)
