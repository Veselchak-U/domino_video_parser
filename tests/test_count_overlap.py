import cv2

from domino_video.vision import ScreenRecognizer


def test_native_retains_unknown_tile_under_counter_without_reading_hidden_pips():
    image = cv2.imread("tests/fixtures/count_overlap1341.png")
    recognizer = ScreenRecognizer()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    normal = recognizer._stones(image, hsv)
    native = recognizer._stones(image, hsv, read_motion=True)
    hidden = [tile for tile in native if 1180 < tile.box[0] < 1200 and 130 < tile.box[1] < 150]
    assert len(hidden) == 1
    assert hidden[0].stone is None
    neighbor = [tile for tile in native if tile.box[0] == 1280 and tile.box[1] == 138]
    assert len(neighbor) == 1
    assert neighbor[0].stone is None
    assert not any(1245 < tile.box[0] < 1299 and tile.box[1] < 130 for tile in native)
    assert [tile for tile in native if tile.box[0] > 1350 and tile.box[1] > 130] == [
        tile for tile in normal if tile.box[0] > 1350 and tile.box[1] > 130
    ]


def test_reveal_does_not_mask_playing_screen_count_regions():
    image = cv2.imread("tests/fixtures/frame_84.png")
    recognizer = ScreenRecognizer()
    prepared = recognizer.prepare(image, 84, False, read_motion=True)
    obs = prepared.finish(recognizer._read, recognizer._read_name)
    assert obs.reveal
    assert [sorted(hand) for hand in obs.hands] == [
        ["2-3"],
        [],
        ["1-5", "2-2", "4-5", "5-5"],
        ["1-2"],
    ]
