"""Local OCR models and adaptive isolated GPU workers."""

import sys
import threading
from pathlib import Path

from .gpu_memory import probe_memory
from .gpu_worker import GPUWorker
from .ocr_result import OCRResult


class DirectMLAdapter:
    @staticmethod
    def available():
        if sys.platform != "win32":
            return False
        import onnxruntime as ort

        return "DmlExecutionProvider" in ort.get_available_providers()

    def __init__(self, profile_prefix=None):
        import numpy as np
        import onnxruntime as ort
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        self._sessions = []
        models = Path(rapidocr_onnxruntime.__file__).parent / "models"
        stages = [
            (self._engine.text_det.infer, "ch_PP-OCRv4_det_infer.onnx", (1, 3, 64, 64)),
            (self._engine.text_cls.infer, "ch_ppocr_mobile_v2.0_cls_infer.onnx", (1, 3, 48, 192)),
            (self._engine.text_rec.session, "ch_PP-OCRv4_rec_infer.onnx", (1, 3, 48, 320)),
        ]
        try:
            for wrapper, filename, shape in stages:
                options = ort.SessionOptions()
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                if profile_prefix:
                    options.enable_profiling = True
                    options.profile_file_prefix = str(profile_prefix) + "-" + filename
                session = ort.InferenceSession(
                    str(models / filename),
                    sess_options=options,
                    providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
                )
                if session.get_providers()[0] != "DmlExecutionProvider":
                    raise RuntimeError(f"Модель {filename} не использует GPU DirectML")
                session.disable_fallback()
                session.run(None, {session.get_inputs()[0].name: np.ones(shape, dtype=np.float32)})
                wrapper.session = session
                self._sessions.append(session)
        except BaseException:
            self.close()
            raise

    def probe(self):
        import numpy as np

        for session, shape in zip(
            self._sessions, [(1, 3, 64, 64), (1, 3, 48, 192), (1, 3, 48, 320)]
        ):
            session.run(None, {session.get_inputs()[0].name: np.ones(shape, dtype=np.float32)})

    def text(self, crop):
        return self.read(crop).text

    def read(self, crop):
        rows, _ = self._engine(crop)
        return OCRResult.from_rows(rows)

    def close(self):
        for session in self._sessions:
            session.end_profiling()
        self._sessions.clear()
        self._engine = None


class DeviceOCR:
    def __init__(
        self, mode, gpu_workers="auto", *, available=None, worker_factory=None, memory_probe=None
    ):
        self.mode = mode
        self._setting = gpu_workers
        self._available = available or DirectMLAdapter.available
        self._factory = worker_factory or GPUWorker
        self._memory = memory_probe or probe_memory
        self._workers = []
        self._loads = []
        self._lock = threading.Lock()
        self.enabled = False
        self.description = "OCR: CPU"

    def _memory_reason(self, threshold):
        snapshot = self._memory()
        if snapshot.free_mib is None:
            return snapshot.reason or "свободная видеопамять неизвестна"
        if snapshot.free_mib < threshold:
            return f"свободно {snapshot.free_mib} MiB, для двух требуется {threshold} MiB"
        return ""

    def __enter__(self):
        if self.mode == "cpu":
            return self
        try:
            if not self._available():
                raise RuntimeError("DirectML недоступен в этой ОС или среде Python")
            reason = "задано --gpu-workers 1" if self._setting == "1" else self._memory_reason(2000)
            self._workers.append(self._factory())
            if not reason:
                reason = self._memory_reason(1152)
            if not reason:
                try:
                    self._workers.append(self._factory())
                    reason = self._memory_reason(256)
                except Exception as error:
                    reason = f"второй GPU-процесс недоступен: {error}"
                if reason:
                    if len(self._workers) == 2:
                        self._workers.pop().close()
                    self._workers[0].probe()
            self._loads = [0] * len(self._workers)
            self.enabled = True
            self.description = f"OCR: GPU (DirectML, адаптер 0), процессов: {len(self._workers)}"
            if reason:
                self.description += f" — {reason}"
        except BaseException as error:
            self.__exit__(None, None, None)
            if not isinstance(error, Exception):
                raise
            if self.mode == "gpu":
                raise RuntimeError(f"Не удалось включить GPU: {error}") from error
            self.description = f"OCR: CPU — GPU недоступен: {error}"
        return self

    def submit(self, prepared):
        with self._lock:
            index = min(range(len(self._workers)), key=self._loads.__getitem__)
            self._loads[index] += 1
        try:
            future = self._workers[index].submit(prepared)
        except BaseException:
            self._completed(index)
            raise
        future.add_done_callback(lambda _: self._completed(index))
        return future

    def _completed(self, index):
        with self._lock:
            self._loads[index] -= 1

    def __exit__(self, *args):
        try:
            for worker in self._workers:
                worker.close()
        finally:
            self._workers.clear()
            self.enabled = False


def main():
    with DeviceOCR("gpu", "1"):
        print("GPU DirectML: all OCR models verified")


if __name__ == "__main__":
    main()
