import argparse
import multiprocessing
import os
import signal
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing

import cv2

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


def _observe_frame(image, timestamp, read_text):
    return _recognizer.observe(image, timestamp, read_text)


class ObservationPipeline:
    def __init__(self, workers=1, recognizer=None):
        self.workers = workers
        self._recognizer = recognizer

    def observe(self, frames):
        cv2.setNumThreads(1)
        with closing(frames):
            if self.workers == 1:
                recognizer = self._recognizer or ScreenRecognizer()
                for image, timestamp, read_text in self._jobs(frames):
                    yield recognizer.observe(image, timestamp, read_text)
                return
            pending = {}
            ready = {}
            submitted = emitted = 0
            exhausted = False
            executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            )
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
                        pending[executor.submit(_observe_frame, *job)] = submitted
                        submitted += 1
                    if not pending:
                        break
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        ready[pending.pop(future)] = future.result()
                    # Buffer only observations, never decoded images. The window
                    # bounds memory even if an early frame takes much longer.
                    while emitted in ready:
                        yield ready.pop(emitted)
                        emitted += 1
            finally:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

    def _jobs(self, frames):
        last_text = -5
        for timestamp, image in frames:
            read_text = timestamp - last_text >= 5
            if read_text:
                last_text = timestamp
            yield image, timestamp, read_text
