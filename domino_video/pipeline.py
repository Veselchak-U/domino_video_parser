import argparse
import multiprocessing
import os
import signal
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import closing
from multiprocessing.util import Finalize

import cv2

from .ocr import DeviceOCR
from .recognition_plan import OCRFields
from .vision import ScreenRecognizer


class WorkerSettings:
    def default(self):
        if hasattr(os, "process_cpu_count"):
            count = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):
            count = len(os.sched_getaffinity(0))
        else:
            count = os.cpu_count()
        return min(count or 1, 61) if sys.platform == "win32" else max(1, count or 1)

    def parse(self, value):
        try:
            count = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Число процессов должно быть целым") from None
        if count < 1 or (sys.platform == "win32" and count > 61):
            raise argparse.ArgumentTypeError(
                "Число процессов должно быть положительным; на Windows не больше 61"
            )
        return count


_recognizer = None


def _initialize_worker():
    global _recognizer
    # Only the parent handles Ctrl+C and closes its bounded queue.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cv2.setNumThreads(1)
    _recognizer = ScreenRecognizer()
    Finalize(_recognizer, _recognizer.close, exitpriority=10)


def _observe_frame(image, timestamp, read_text):
    return _recognizer.observe(image, timestamp, read_text)


def _prepare_frame(image, timestamp, read_text):
    return _recognizer.prepare(image, timestamp, read_text)


def _motion_observation(recognizer, image, timestamp):
    result = recognizer.prepare(image, timestamp, False, read_motion=True).observation
    result.dense = True
    return result


def _motion_frame(image, timestamp, read_text):
    return _motion_observation(_recognizer, image, timestamp)


class ObservationPipeline:
    def __init__(
        self,
        workers=1,
        recognizer=None,
        device="cpu",
        message=None,
        gpu_workers="auto",
        fields: dict[float, OCRFields] | None = None,
        motion=False,
    ):
        self.workers = workers
        self._recognizer = recognizer
        self._device = device
        self._message = message
        self._gpu_workers = gpu_workers
        self._fields = fields
        self._motion = motion

    def observe(self, frames):
        if self._motion:
            yield from self._observe(frames, None)
            return
        with closing(frames), DeviceOCR(self._device, self._gpu_workers) as gpu:
            if self._message:
                self._message(gpu.description)
            yield from self._observe(frames, gpu)

    def _observe(self, frames, gpu):
        cv2.setNumThreads(1)
        gpu_enabled = gpu is not None and gpu.enabled
        with closing(frames):
            if self.workers == 1 and not gpu_enabled:
                recognizer = self._recognizer or ScreenRecognizer()
                try:
                    for image, timestamp, read_text in self._jobs(frames):
                        yield (
                            _motion_observation(recognizer, image, timestamp)
                            if self._motion
                            else recognizer.observe(image, timestamp, read_text)
                        )
                finally:
                    if hasattr(recognizer, "close"):
                        recognizer.close()
                return
            pending = {}
            ready = {}
            submitted = emitted = 0
            exhausted = False
            executor = (
                ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                )
                if self.workers > 1
                else None
            )
            gpu_pending = set()
            recognizer = self._recognizer or ScreenRecognizer()
            try:
                jobs = iter(self._jobs(frames))
                while pending or not exhausted:
                    while (
                        not exhausted
                        and len(pending) < 2 * self.workers
                        and submitted - emitted < 32 * self.workers
                    ):
                        job = next(jobs, None)
                        if job is None:
                            exhausted = True
                            break
                        if executor is None:
                            future = Future()
                            future.set_result(recognizer.prepare(*job))
                        else:
                            work = (
                                _motion_frame
                                if self._motion
                                else _prepare_frame
                                if gpu_enabled
                                else _observe_frame
                            )
                            future = executor.submit(work, *job)
                        pending[future] = submitted
                        submitted += 1
                    if not pending:
                        break
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        number = pending.pop(future)
                        result = future.result()
                        if gpu_enabled and future not in gpu_pending and result.crops:
                            following = gpu.submit(result)
                            pending[following] = number
                            gpu_pending.add(following)
                        else:
                            ready[number] = (
                                result.observation
                                if gpu_enabled and future not in gpu_pending
                                else result
                            )
                        gpu_pending.discard(future)
                    # Buffer only observations, never decoded images. The window
                    # bounds memory even if an early frame takes much longer.
                    while emitted in ready:
                        yield ready.pop(emitted)
                        emitted += 1
            finally:
                if hasattr(recognizer, "close"):
                    recognizer.close()
                for future in pending:
                    future.cancel()
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

    def _jobs(self, frames):
        last_text = -5
        for timestamp, image in frames:
            read_text = timestamp - last_text >= 5
            if read_text:
                last_text = timestamp
            if self._fields is not None:
                read_text = self._fields.get(timestamp, OCRFields())
            yield image, timestamp, read_text
