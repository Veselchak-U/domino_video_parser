import pytest


class ThreadWorker:
    """In-process worker double for pipeline scheduling tests, without native OCR."""

    def __init__(self, factory):
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.adapter = self.executor.submit(factory).result()
        except BaseException:
            self.executor.shutdown()
            raise

    def submit(self, prepared):
        return self.executor.submit(prepared.finish, self.adapter.read)

    def probe(self):
        pass

    def close(self):
        self.executor.submit(self.adapter.close).result()
        self.executor.shutdown(cancel_futures=True)


@pytest.mark.parametrize("mode", ["auto", "cpu", "gpu"])
@pytest.mark.parametrize("available", [False, True])
def test_device_selection(mode, available):
    from domino_video.ocr import DeviceOCR

    calls = []

    class Adapter:
        def __init__(self):
            calls.append("init")

        def close(self):
            calls.append("close")

    session = DeviceOCR(
        mode,
        available=lambda: available,
        worker_factory=lambda: ThreadWorker(Adapter),
        gpu_workers="1",
    )
    if mode == "gpu" and not available:
        with pytest.raises(RuntimeError, match="GPU"):
            with session:
                pass
    else:
        with session:
            assert session.enabled == (available and mode != "cpu")
        assert calls == (["init", "close"] if available and mode != "cpu" else [])


@pytest.mark.parametrize("mode", ["auto", "gpu"])
def test_failed_gpu_initialization_is_not_reported_as_gpu(mode):
    from domino_video.ocr import DeviceOCR

    def fail():
        raise RuntimeError("model failed")

    session = DeviceOCR(mode, available=lambda: True, worker_factory=fail, gpu_workers="1")
    if mode == "gpu":
        with pytest.raises(RuntimeError, match="model failed"):
            with session:
                pass
    else:
        with session:
            assert not session.enabled
            assert "model failed" in session.description


def test_cli_rejects_unknown_device():
    from domino_video.cli import main

    with pytest.raises(SystemExit) as error:
        main(["--device", "invalid", "--fish-variant", "withoutEggs", "--score-limit", "50"])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["auto", "cpu", "gpu"])
def test_cli_passes_device(tmp_path, monkeypatch, mode):
    from domino_video.cli import main

    (tmp_path / "video.mp4").touch()
    calls = []
    monkeypatch.setattr(
        "domino_video.manager.ParseManager.run", lambda self, *args: calls.append(args) or 0
    )
    assert (
        main(
            [
                "--input",
                str(tmp_path),
                "--device",
                mode,
                "--fish-variant",
                "withoutEggs",
                "--score-limit",
                "50",
            ]
        )
        == 0
    )
    assert calls[0][-2] == mode


@pytest.mark.parametrize(
    "platform,providers,expected",
    [
        ("linux", ["DmlExecutionProvider"], False),
        ("win32", ["CPUExecutionProvider"], False),
        ("win32", ["DmlExecutionProvider", "CPUExecutionProvider"], True),
    ],
)
def test_gpu_availability_checks_os_and_provider(monkeypatch, platform, providers, expected):
    from domino_video.ocr import DirectMLAdapter

    monkeypatch.setattr("domino_video.ocr.sys.platform", platform)
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: providers)
    assert DirectMLAdapter.available() is expected


def test_model_silently_using_cpu_is_rejected(monkeypatch):
    from domino_video.ocr import DirectMLAdapter

    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", lambda **kwargs: engine)
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda *args, **kwargs: Session())
    with pytest.raises(RuntimeError, match="не использует GPU"):
        DirectMLAdapter()


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_gpu_pipeline_order_bounds_owner_and_cleanup(monkeypatch, workers, failure):
    import threading
    from concurrent.futures import Future

    from domino_video.ocr import DeviceOCR
    from domino_video.pipeline import ObservationPipeline

    owner = []
    completed = []
    closed = []
    baseline = set(threading.enumerate())

    class Adapter:
        def __init__(self):
            owner.append(threading.get_ident())

        def text(self, number):
            assert threading.get_ident() == owner[0]
            if failure:
                raise failure("GPU failed")
            completed.append(number)
            return number

        read = text

        def close(self):
            assert threading.get_ident() == owner[0]
            closed.append("gpu")

    class Prepared:
        crops = {"text": [1]}

        def __init__(self, number):
            self.number = number

        def finish(self, text):
            return text(self.number)

    class Recognizer:
        def prepare(self, image, timestamp, read_text):
            return Prepared(timestamp)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, image, timestamp, read_text):
            future = Future()
            future.set_result(Prepared(timestamp))
            return future

        def shutdown(self, **kwargs):
            closed.append("cpu")

    monkeypatch.setattr("domino_video.pipeline.ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        "domino_video.pipeline.DeviceOCR",
        lambda mode, setting: DeviceOCR(
            mode,
            available=lambda: True,
            worker_factory=lambda: ThreadWorker(Adapter),
            gpu_workers="1",
        ),
    )

    def frames():
        try:
            for number in range(40):
                assert number - len(completed) < 2 * workers
                yield number, None
        finally:
            closed.append("input")

    iterator = ObservationPipeline(workers, Recognizer(), device="gpu").observe(frames())
    if failure:
        with pytest.raises(failure, match="GPU failed"):
            list(iterator)
    else:
        assert list(iterator) == list(range(40))
    assert closed.count("gpu") == 1
    assert closed.count("input") == 1
    assert len(owner) == 1 and owner[0] != threading.get_ident()
    assert set(threading.enumerate()) == baseline


@pytest.mark.parametrize(
    "name,read_text,expected", [("20", True, 9), ("20", False, 0), ("84", True, 2)]
)
def test_prepared_crops_preserve_observations(name, read_text, expected):
    import cv2
    import numpy as np

    from domino_video.ocr_result import OCRLine, OCRResult
    from domino_video.vision import ScreenRecognizer

    image = cv2.imread(f"tests/fixtures/frame_{name}.png")
    recognizer = ScreenRecognizer()
    prepared = recognizer.prepare(image, float(name), read_text)
    assert sum(map(len, prepared.crops.values())) == expected
    for group in prepared.crops.values():
        for crop in group:
            assert not np.shares_memory(crop, image)
    reading = OCRResult((OCRLine("0/50", 0.99),))
    actual = prepared.finish(lambda crop: reading)
    recognizer._read = lambda crop: reading
    recognizer._read_name = lambda crop: reading
    assert actual == recognizer.observe(image, float(name), read_text)


def test_gpu_failure_does_not_break_next_source(tmp_path, monkeypatch, observations):
    import json
    from copy import deepcopy

    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager
    from domino_video.ocr import DeviceOCR
    from domino_video.video import VideoTimeline

    initialized = []

    class Adapter:
        def __init__(self):
            self.number = len(initialized)
            initialized.append(self)
            self.closed = False

        def text(self, crop):
            if self.number == 0:
                raise RuntimeError("GPU memory exhausted")

        read = text

        def close(self):
            self.closed = True

    class Prepared:
        crops = {"text": [1]}

        def __init__(self, observation):
            self.observation = observation

        def finish(self, text):
            text(None)
            return self.observation

    class Recognizer:
        def observe(self, image, time, read_text):
            visual = deepcopy(image)
            visual.selective = True
            return visual

        def prepare(self, image, time, read_text):
            return Prepared(image)

    class Reader:
        def frame_at(self, path, time):
            return next(o for o in observations if o.time == time)

        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            for observation in observations:
                yield observation.time, observation

    monkeypatch.setattr(
        "domino_video.pipeline.DeviceOCR",
        lambda mode, setting: DeviceOCR(
            mode,
            available=lambda: True,
            worker_factory=lambda: ThreadWorker(Adapter),
            gpu_workers="1",
        ),
    )
    sources = [tmp_path / f"2026-09-28-13-{minute}.mp4" for minute in (58, 59)]
    for source in sources:
        source.write_bytes(b"fixture")
    output = tmp_path / "out"
    assert (
        ParseManager(Reader(), Recognizer()).run(
            sources, output, "withoutEggs", 50, Corrections(), 1, "gpu"
        )
        == 1
    )
    reports = sorted((output / "report").glob("*.json"))
    assert [json.loads(path.read_text(encoding="utf-8"))["status"] for path in reports] == [
        "error",
        "ok",
    ]
    assert len(list(output.glob("*game*.json"))) == 1
    assert len(initialized) == 2 and all(adapter.closed for adapter in initialized)


def test_unscheduled_empty_table_still_reads_scores(monkeypatch):
    import cv2

    from domino_video.ocr_result import OCRLine, OCRResult
    from domino_video.vision import ScreenRecognizer

    recognizer = ScreenRecognizer()
    monkeypatch.setattr(recognizer, "_stones", lambda *args: [])
    prepared = recognizer.prepare(cv2.imread("tests/fixtures/frame_20.png"), 1, False)
    assert set(prepared.crops) == {"scores"}
    result = prepared.finish(lambda crop: OCRResult((OCRLine("7/50", 0.99),)))
    assert result.scores == (7, 7) and result.limit == 50


def test_gpu_window_waits_for_early_cpu_frame_without_losing_order(monkeypatch):
    from concurrent.futures import Future
    from concurrent.futures import wait as real_wait

    from domino_video import pipeline

    submitted, emitted = [], []
    first = None

    class Prepared:
        crops = {"text": [1]}

        def __init__(self, number):
            self.number = number

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, image, timestamp, read_text):
            nonlocal first
            future = Future()
            submitted.append(timestamp)
            assert len(submitted) - len(emitted) <= 64
            if timestamp == 0:
                first = future
            else:
                future.set_result(Prepared(timestamp))
            return future

        def shutdown(self, **kwargs):
            pass

    class GPU:
        enabled = True

        def __init__(self, mode, setting):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, prepared):
            future = Future()
            future.set_result(prepared.number)
            return future

    def controlled_wait(pending, **kwargs):
        assert len(pending) <= 4
        if not any(f.done() for f in pending):
            assert len(submitted) == 64
            first.set_result(Prepared(0))
        return real_wait(pending, **kwargs)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(pipeline, "DeviceOCR", GPU)
    monkeypatch.setattr(pipeline, "wait", controlled_wait)
    for result in pipeline.ObservationPipeline(2, device="gpu").observe(
        (t, None) for t in range(100)
    ):
        emitted.append(result)
    assert emitted == list(range(100))
