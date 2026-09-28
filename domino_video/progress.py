import os
import sys
import threading
import time


class ConsoleProgress:
    def __init__(self, stream=None, clock=time.monotonic):
        self._stream = stream if stream is not None else sys.stdout
        self._interactive = self._stream.isatty()
        self._color = self._interactive and self._enable_color()
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._running = False
        self._started = 0.0
        self._percent = 0
        self._processed_seconds = 0.0
        self._speed = 0.0
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
            self._processed_seconds = 0.0
            self._speed = 0.0
            self._phase = ""
            self._last = None
            self._width = 0
            self._stop.clear()
            self._stream.write(f"[{index}/{total}] {name}\n")
            if self._interactive:
                self._draw("  Обработано 0% за 0 сек скорость 0x")
            self._stream.flush()
            self._running = True
        if self._interactive:
            self._thread = threading.Thread(target=self._refresh, name="domino-progress")
            self._thread.start()

    def update(self, percent, phase="", *, processed_seconds=None):
        with self._lock:
            if processed_seconds is not None:
                self._processed_seconds = max(self._processed_seconds, processed_seconds)
            percent = max(self._percent, min(99, int(percent)))
            if percent != self._percent:
                self._update_speed(max(0, self._clock() - self._started))
            self._percent = percent
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
            elapsed = max(0, self._clock() - self._started)
            if success:
                self._update_speed(elapsed)
            self._running = False
            self._stop.set()
        # Joining while holding the output lock would deadlock a pending refresh.
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            percent = 100 if success else self._percent
            text = self._status(percent, int(elapsed), "")
            self._draw(text, error=not success)
            self._end_line()

    def _enable_color(self):
        if os.name != "nt":
            return os.environ.get("TERM") != "dumb"
        import ctypes
        import msvcrt
        from ctypes import wintypes

        try:
            handle = wintypes.HANDLE(msvcrt.get_osfhandle(self._stream.fileno()))
            mode = wintypes.DWORD()
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING, preserving other console flags.
            return bool(kernel.SetConsoleMode(handle, wintypes.DWORD(mode.value | 0x0004)))
        except (OSError, ValueError, AttributeError):
            return False

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
                text = self._status(self._percent, seconds, self._phase)
                if text != self._last:
                    self._draw(text)

    def _update_speed(self, elapsed):
        self._speed = self._processed_seconds / elapsed if elapsed > 0 else 0.0

    def _status(self, percent, seconds, phase):
        text = f"  Обработано {percent}% за {seconds} сек"
        if phase:
            text += f" — {phase}"
        speed = f"{self._speed:.2f}".rstrip("0").rstrip(".")
        return f"{text} скорость {speed}x"

    def _draw(self, text, *, error=False):
        suffix = " — ошибка" if error else ""
        width = len(text) + len(suffix)
        if suffix and self._color:
            suffix = f"\x1b[31m{suffix}\x1b[39m"
        if self._active:
            self._stream.write("\r")
        self._stream.write(text + suffix + " " * max(0, self._width - width))
        self._stream.flush()
        self._width = width
        self._last = text
        self._active = True

    def _end_line(self):
        if self._active:
            self._stream.write("\n")
            self._stream.flush()
            self._active = False
            self._width = 0
