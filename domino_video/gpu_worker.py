"""An isolated OCR process with bounded startup and cancellation."""

import multiprocessing
import queue
import signal
import threading
from concurrent.futures import Future


def _serve(connection, factory):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    adapter = None
    try:
        import cv2

        cv2.setNumThreads(1)
        adapter = factory()
        connection.send((True, None))
        while True:
            command = connection.recv()
            if command is None:
                return
            kind, value = command
            if kind == "probe":
                result = adapter.probe()
            elif hasattr(adapter, "read_name"):
                result = value.finish(adapter.read, adapter.read_name)
            else:
                result = value.finish(adapter.read)
            connection.send((True, result))
    except BaseException as error:
        try:
            connection.send((False, f"{type(error).__name__}: {error}"))
        except (OSError, EOFError):
            pass
    finally:
        if adapter is not None:
            adapter.close()
        connection.close()


class GPUWorker:
    def __init__(self, factory=None, *, startup_timeout=60, stop_timeout=5):
        if factory is None:
            from .ocr import DirectMLAdapter

            factory = DirectMLAdapter
        self._startup_timeout = startup_timeout
        self._stop_timeout = stop_timeout
        self._stop = threading.Event()
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._failure = None
        self.closed = False
        self._ready = Future()
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child, factory), name="domino-gpu")
        self._thread = threading.Thread(target=self._dispatch, name="domino-gpu-ipc")
        try:
            self._process.start()
            child.close()
            self._thread.start()
            self._ready.result(timeout=startup_timeout)
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self):
        while not self._connection.poll(0.1):
            if self._stop.is_set():
                raise RuntimeError("GPU-процесс остановлен")
            if not self._process.is_alive():
                raise RuntimeError(f"GPU-процесс завершился: {self._process.exitcode}")
        try:
            ok, result = self._connection.recv()
        except (EOFError, OSError) as error:
            self._process.join(0.1)
            raise RuntimeError(
                f"Соединение с GPU-процессом потеряно, код: {self._process.exitcode}"
            ) from error
        if not ok:
            raise RuntimeError(result)
        return result

    def _dispatch(self):
        current = self._ready
        try:
            self._receive()
            current.set_result(None)
            current = None
            while not self._stop.is_set():
                try:
                    current, command = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if not self._process.is_alive():
                        raise RuntimeError(f"GPU-процесс завершился: {self._process.exitcode}")
                    continue
                if self._stop.is_set():
                    current.cancel()
                    current = None
                    break
                if not current.set_running_or_notify_cancel():
                    current = None
                    continue
                self._connection.send(command)
                current.set_result(self._receive())
                current = None
            self._connection.send(None)
        except BaseException as error:
            failure = RuntimeError(f"Ошибка GPU-процесса: {error}")
            if current is not None and not current.done():
                current.set_exception(failure)
            with self._lock:
                self._failure = failure
        finally:
            with self._lock:
                self._failure = self._failure or RuntimeError("GPU-процесс остановлен")
                while True:
                    try:
                        future, _ = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if self._stop.is_set():
                        future.cancel()
                    elif not future.done():
                        future.set_exception(self._failure)

    def _submit(self, command):
        future = Future()
        with self._lock:
            if self._failure or self._stop.is_set():
                future.set_exception(self._failure or RuntimeError("GPU-процесс остановлен"))
            else:
                self._queue.put((future, command))
        return future

    def submit(self, prepared):
        return self._submit(("ocr", prepared))

    def probe(self):
        return self._submit(("probe", None)).result(timeout=self._startup_timeout)

    def close(self):
        if self.closed:
            return
        self._stop.set()
        if self._process.pid is not None:
            self._process.join(self._stop_timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        if self._thread.ident is not None:
            self._thread.join()
        self._connection.close()
        self._process.close()
        self.closed = True
