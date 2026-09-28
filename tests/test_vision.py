from pathlib import Path

import cv2
import pytest


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("scale", [1, 0.75])
def test_reads_dominoes_after_rotation_and_scaling(turns, scale):
    from domino_video.vision import ScreenRecognizer

    im = cv2.imread(str(Path("tests/fixtures/frame_20.png")))
    im = cv2.resize(im, None, fx=scale, fy=scale)
    for _ in range(turns):
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    obs = ScreenRecognizer().observe(im, 20)
    assert {s.stone for s in obs.board} == {"1-1", "1-6", "0-1"}
    assert obs.active == 3


def test_reads_revealed_hands():
    from domino_video.vision import ScreenRecognizer

    obs = ScreenRecognizer().observe(cv2.imread("tests/fixtures/frame_84.png"), 84)
    assert obs.reveal
    assert [sorted(h) for h in obs.hands] == [["2-3"], [], ["1-5", "2-2", "4-5", "5-5"], ["1-2"]]


def test_manual_video_reference_satisfies_rules():
    import json

    from domino_video.validator import GameValidator

    game = json.loads(Path("tests/fixtures/sample_game.json").read_text(encoding="utf-8"))
    assert GameValidator().validate(game)["result"] == {"winner": "Team B"}


def test_reads_visible_player_counts_locally(monkeypatch):
    import socket

    from domino_video.vision import ScreenRecognizer

    def no_network(*args, **kwargs):
        raise AssertionError("Network access during OCR")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    obs = ScreenRecognizer().observe(cv2.imread("tests/fixtures/frame_20.png"), 20, True)
    assert obs.counts == (6, 6, 6, 7)
