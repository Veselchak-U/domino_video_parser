import sys
import threading
import time


class ConsoleProgress:
    def __init__(self, stream=None, clock=time.monotonic):
        self._stream = stream if stream is not None else sys.stdout
        self._interactive = self._stream.isatty()
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._running = False
        self._started = 0.0
        self._percent = 0
        self._phase = ""
        self._last = None
        self._width = 0
        self._active = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish(False)

    def start(self, index, total, name):
        self.finish(False)
        with self._lock:
            self._started = self._clock()
            self._percent = 0
            self._phase = ""
            self._last = None
            self._width = 0
            self._stop.clear()
            self._stream.write(f"[{index}/{total}] {name}\n")
            if self._interactive:
                self._draw("  Обработано 0% за 0 сек")
            self._stream.flush()
            self._running = True
        if self._interactive:
            self._thread = threading.Thread(target=self._refresh, name="domino-progress")
            self._thread.start()

    def update(self, percent, phase=""):
        with self._lock:
            self._percent = max(self._percent, min(99, int(percent)))
            self._phase = phase

    def message(self, text):
        with self._lock:
            self._end_line()
            self._stream.write(text + "\n")
            self._stream.flush()

    def finish(self, success):
        with self._lock:
            if not self._running:
                return
            elapsed = int(max(0, self._clock() - self._started))
            self._running = False
            self._stop.set()
        # Joining while holding the output lock would deadlock a pending refresh.
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            percent = 100 if success else self._percent
            text = f"  Обработано {percent}% за {elapsed} сек"
            if not success:
                text += " — ошибка"
            self._draw(text)
            self._end_line()

    def _refresh(self):
        while True:
            with self._lock:
                elapsed = max(0, self._clock() - self._started)
                delay = 1 - (elapsed % 1)
            if self._stop.wait(delay):
                return
            with self._lock:
                if not self._running:
                    return
                seconds = int(max(0, self._clock() - self._started))
                text = f"  Обработано {self._percent}% за {seconds} сек"
                if self._phase:
                    text += f" — {self._phase}"
                if text != self._last:
                    self._draw(text)

    def _draw(self, text):
        if self._active:
            self._stream.write("\r")
        self._stream.write(text + " " * max(0, self._width - len(text)))
        self._stream.flush()
        self._width = len(text)
        self._last = text
        self._active = True

    def _end_line(self):
        if self._active:
            self._stream.write("\n")
            self._stream.flush()
            self._active = False
            self._width = 0
