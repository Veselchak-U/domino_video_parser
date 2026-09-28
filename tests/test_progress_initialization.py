import io

import pytest

from domino_video.progress import ConsoleProgress


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("tty", [False, True])
@pytest.mark.parametrize("backend", ["CPU", "GPU", "CPU — GPU недоступен"])
def test_initialization_does_not_leave_zero_line(tty, backend):
    stream = Terminal() if tty else io.StringIO()
    now = [0]
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(1, 1, "video.mp4")
        now[0] = 5
        progress.update(0, "определение длительности")
        assert stream.getvalue() == "[1/1] video.mp4\n"
        progress.message("OCR: " + backend)
        progress.ready()
        now[0] = 10
        progress.update(50, processed_seconds=100)
        progress.finish(True)
    assert stream.getvalue().startswith("[1/1] video.mp4\nOCR: " + backend + "\n")
    assert "100% за 10 сек скорость 10x" in stream.getvalue()


def test_failed_initialization_still_has_final_status():
    stream = Terminal()
    now = [0]
    with ConsoleProgress(stream, clock=lambda: now[0]) as progress:
        progress.start(1, 1, "video.mp4")
        now[0] = 5
        progress.finish(False)
    assert stream.getvalue().count("Обработано") == 1
    assert "0% за 5 сек скорость 0x" in stream.getvalue()
