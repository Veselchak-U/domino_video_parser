"""Local OCR backend and its single-threaded GPU lifetime."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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

    def text(self, crop):
        rows, _ = self._engine(crop)
        return " ".join(row[1] for row in (rows or []) if row[2] > 0.8)

    def close(self):
        for session in self._sessions:
            session.end_profiling()
        self._sessions.clear()
        self._engine = None


class DeviceOCR:
    def __init__(self, mode, *, available=None, factory=None):
        self.mode = mode
        self._available = available or DirectMLAdapter.available
        self._factory = factory or DirectMLAdapter
        self._executor = None
        self._adapter = None
        self.enabled = False
        self.description = "OCR: CPU"

    def __enter__(self):
        if self.mode == "cpu":
            return self
        try:
            if not self._available():
                raise RuntimeError("DirectML недоступен в этой ОС или среде Python")
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="domino-gpu")
            self._executor.submit(self._initialize).result()
            self.enabled = True
            self.description = "OCR: GPU (DirectML, адаптер 0)"
        except BaseException as error:
            self.__exit__(None, None, None)
            if not isinstance(error, Exception):
                raise
            if self.mode == "gpu":
                raise RuntimeError(f"Не удалось включить GPU: {error}") from error
            self.description = f"OCR: CPU — GPU недоступен: {error}"
        return self

    def _initialize(self):
        self._adapter = self._factory()

    def submit(self, prepared):
        return self._executor.submit(prepared.finish, self._adapter.text)

    def __exit__(self, *args):
        if self._executor is not None:
            try:
                self._executor.submit(self._close).result()
            finally:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
        self.enabled = False

    def _close(self):
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
