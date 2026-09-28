import io
import threading

import pytest

from domino_video.progress import ConsoleProgress


class Terminal(io.StringIO):
    def __init__(self):
        super().__init__()
        self.changed = threading.Condition()

    def isatty(self):
        return True

    def write(self, value):
        with self.changed:
            result = super().write(value)
            self.changed.notify_all()
            return result

    def wait_for(self, text):
        with self.changed:
            assert self.changed.wait_for(lambda: text in self.getvalue(), timeout=5)


@pytest.mark.parametrize("phase", ["", "определение длительности", "проверка и сохранение"])
def test_timer_updates_without_new_observations_and_stops(phase):
    stream = Terminal()
    now = [100.0]
    baseline = set(threading.enumerate())
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(2, 5, "длинное_имя_" * 30 + ".mp4")
        progress.ready()
        progress.update(10.9, phase)
        progress.update(5, phase)
        assert "Обработано 10%" not in stream.getvalue()
        now[0] = 115.9
        stream.wait_for("  Обработано 10% за 15 сек" + (f" — {phase}" if phase else ""))
        now[0] = 116.1
        stream.wait_for("  Обработано 10% за 16 сек")
        progress.message("Служебное сообщение")
        now[0] = 117.9
        progress.finish(True)
    text = stream.getvalue()
    assert text.startswith("[2/5] ")
    assert text.count(".mp4") == 1
    assert "\n  Обработано 0% за 0 сек" in text
    assert "\nСлужебное сообщение\n" in text
    assert text.endswith("  Обработано 100% за 17 сек скорость 0x\n")
    assert set(threading.enumerate()) == baseline


@pytest.mark.parametrize("terminal", [False, True])
def test_final_time_is_immediate_truncated_and_reset_for_next_file(terminal):
    stream = Terminal() if terminal else io.StringIO()
    now = [10.0]
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(1, 2, "first.mp4")
        progress.ready()
        now[0] = 145.9
        progress.finish(True)
        progress.start(2, 2, "second.mp4")
        progress.ready()
        now[0] = 146.3
        progress.finish(True)
    text = stream.getvalue()
    assert "Обработано 100% за 135 сек" in text
    assert "Обработано 100% за 0 сек" in text
    if not terminal:
        assert "\r" not in text
        assert len(text.splitlines()) == 4


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_context_stops_timer_on_error_or_interrupt(failure, monkeypatch):
    monkeypatch.setattr(ConsoleProgress, "_enable_color", lambda self: False)
    stream = Terminal()
    now = [0.0]
    baseline = set(threading.enumerate())
    with pytest.raises(failure):
        with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
            progress.start(1, 1, "video.mp4")
            progress.ready()
            progress.update(100)
            now[0] = 9.8
            raise failure("stop")
    assert "100%" not in stream.getvalue()
    assert "  Обработано 99% за 9 сек скорость 0x — ошибка" in stream.getvalue()
    assert set(threading.enumerate()) == baseline


def test_redirected_output_has_no_timer_or_intermediate_lines():
    stream = io.StringIO()
    now = [0.0]
    baseline = set(threading.enumerate())
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(1, 1, "video.mp4")
        progress.ready()
        for percent in range(100):
            now[0] = percent
            progress.update(percent)
        assert set(threading.enumerate()) == baseline
        assert stream.getvalue() == "[1/1] video.mp4\n"
        progress.finish(False)
    assert stream.getvalue() == "[1/1] video.mp4\n  Обработано 99% за 99 сек скорость 0x — ошибка\n"


@pytest.mark.parametrize("terminal,color", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_error_suffix_color_order_and_frozen_speed(monkeypatch, terminal, color, failure):
    monkeypatch.setattr(ConsoleProgress, "_enable_color", lambda self: color, raising=False)
    stream = Terminal() if terminal else io.StringIO()
    now = [0.0]
    with pytest.raises(failure):
        with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
            progress.start(1, 2, "first.mp4")
            progress.ready()
            now[0] = 5
            progress.update(99, processed_seconds=100)
            now[0] = 10
            raise failure()
    suffix = "\x1b[31m — ошибка\x1b[39m" if terminal and color else " — ошибка"
    assert stream.getvalue().endswith(f"  Обработано 99% за 10 сек скорость 20x{suffix}\n")
    if not terminal or not color:
        assert "\x1b" not in stream.getvalue()
    assert "100%" not in stream.getvalue()
    progress.message("Обычный текст")
    assert stream.getvalue().endswith(suffix + "\nОбычный текст\n")


def test_colored_error_clears_longer_previous_phase(monkeypatch):
    monkeypatch.setattr(ConsoleProgress, "_enable_color", lambda self: True)
    stream = Terminal()
    now = [0.0]
    phase = "проверка и сохранение"
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(1, 1, "video.mp4")
        progress.ready()
        now[0] = 5
        progress.update(99, phase, processed_seconds=100)
        before = f"  Обработано 99% за 5 сек — {phase} скорость 20x"
        stream.wait_for(before)
        progress.finish(False)
    plain = "  Обработано 99% за 5 сек скорость 20x — ошибка"
    padding = " " * (len(before) - len(plain))
    assert stream.getvalue().endswith("\x1b[31m — ошибка\x1b[39m" + padding + "\n")


@pytest.mark.parametrize("interrupt", [False, True])
def test_manager_keeps_timer_during_decode_and_report_write(
    tmp_path, monkeypatch, observations, interrupt
):
    from domino_video.corrections import Corrections
    from domino_video.manager import ParseManager
    from domino_video.video import VideoTimeline

    stream = Terminal()
    now = [0.0]
    baseline = set(threading.enumerate())
    progress = ConsoleProgress(stream, clock=lambda: now[0])
    monkeypatch.setattr("domino_video.manager.ConsoleProgress", lambda: progress)

    class Reader:
        def timeline(self, path, scanning=None):
            return VideoTimeline(0, 291)

        def frames(self, path):
            now[0] = 2.0
            stream.wait_for("Обработано 0% за 2 сек")
            for item in observations:
                yield item.time, item

    class Recognizer:
        def observe(self, image, timestamp, read_text):
            return image

    manager = ParseManager(Reader(), Recognizer())
    original = manager._storage.write

    def write(path, value, **kwargs):
        if path.name.endswith("-report.json"):
            now[0] = 4.0
            stream.wait_for("Обработано 99% за 4 сек — проверка и сохранение")
            assert "100%" not in stream.getvalue()
            if interrupt:
                raise KeyboardInterrupt
        original(path, value, **kwargs)

    monkeypatch.setattr(manager._storage, "write", write)
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"fixture")
    code = manager.run([source], tmp_path / "out", "withoutEggs", 50, Corrections())
    assert code == int(interrupt)
    assert ("Обработано 100% за 4 сек" in stream.getvalue()) == (not interrupt)
    assert set(threading.enumerate()) == baseline


def test_speed_updates_only_with_percent_and_resets():
    stream = Terminal()
    now = [0.0]
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(1, 2, "first.mp4")
        progress.ready()
        now[0] = 5
        progress.update(10, processed_seconds=100)
        stream.wait_for("Обработано 10% за 5 сек скорость 20x")
        now[0] = 6
        progress.update(10.9, processed_seconds=109)
        stream.wait_for("Обработано 10% за 6 сек скорость 20x")
        now[0] = 10
        progress.update(11, processed_seconds=110)
        stream.wait_for("Обработано 11% за 10 сек скорость 11x")
        progress.update(99, processed_seconds=200)
        now[0] = 20
        progress.finish(True)
        assert "Обработано 100% за 20 сек скорость 10x" in stream.getvalue()
        progress.start(2, 2, "second.mp4")
        progress.ready()
        progress.update(10, processed_seconds=100)
        progress.finish(True)
    assert stream.getvalue().rstrip().endswith("Обработано 100% за 0 сек скорость 0x")


@pytest.mark.parametrize("seconds,expected", [(3, "3.33x"), (4, "2.5x"), (5, "2x")])
def test_speed_format(seconds, expected):
    stream = io.StringIO()
    now = [0.0]
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(1, 1, "video.mp4")
        progress.ready()
        now[0] = seconds
        progress.update(99, processed_seconds=10)
        progress.finish(True)
    assert stream.getvalue().rstrip().endswith("скорость " + expected)
