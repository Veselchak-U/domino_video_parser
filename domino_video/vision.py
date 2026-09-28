"""Визуальный профиль записи: координаты нормализованы к 1608×720."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StoneObservation:
    values: tuple[int, int]
    box: tuple[int, int, int, int]

    @property
    def stone(self):
        return "-".join(map(str, sorted(self.values)))

    @property
    def center(self):
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


@dataclass
class Observation:
    time: float
    board: list[StoneObservation]
    hands: list[list[str]]
    active: int | None
    reveal: bool
    supported: bool
    names: list[str] | None = None
    scores: tuple[int, int] | None = None
    limit: int | None = None
    counts: tuple[int | None, ...] | None = None


class ScreenRecognizer:
    def __init__(self):
        self._ocr = None

    def normalize(self, image):
        candidates = []
        for turns in range(4):
            im = np.rot90(image, turns)
            if im.shape[1] < im.shape[0]:
                continue
            im = cv2.resize(im, (1608, 720))
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            # Three blue action buttons are on the right in the supported profile.
            blue = cv2.inRange(
                hsv[230:425, 1490:1590], np.array([85, 140, 100]), np.array([115, 255, 255])
            )
            candidates.append((cv2.countNonZero(blue), im))
        return max(candidates, key=lambda pair: pair[0])[1]

    def observe(self, image, time, read_text=False):
        im = self.normalize(image)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        table = hsv[150:550, 200:1450]
        green = cv2.inRange(table, np.array([35, 100, 20]), np.array([100, 255, 255]))
        supported = cv2.countNonZero(green) > 150000
        level = float(np.median(table[:, :, 2][green > 0])) if supported else 0
        reveal = supported and level < 80
        tiles = self._stones(im, hsv)
        board = []
        hands = [[], [], [], []]
        for tile in tiles:
            x, y, w, h = tile.box
            if reveal:
                if 45 < y < 150 and h > w and 100 < x < 1450:
                    seat = 1 if x < 500 else (2 if x < 1050 else 3)
                    hands[seat].append(tile.stone)
                elif 390 < y < 470 and h > w and 500 < x < 1150:
                    hands[0].append(tile.stone)
            elif y > 560 and 450 < x < 1350:
                hands[0].append(tile.stone)
            elif 140 < y < 550 and 180 < x < 1480:
                board.append(tile)
        intensity = []
        for x, y in [(188, 538), (157, 80), (691, 80), (1224, 80)]:
            crop = hsv[y - 45 : y + 46, x - 45 : x + 46]
            intensity.append(
                cv2.countNonZero(
                    cv2.inRange(crop, np.array([8, 120, 170]), np.array([30, 255, 255]))
                )
            )
        active = int(np.argmax(intensity)) if max(intensity) > 500 and not reveal else None
        result = Observation(time, board, hands, active, reveal, supported)
        if read_text and supported and not reveal:
            result.names = self._read_names(im)
            result.counts = self._read_counts(im, len(hands[0]))
        if supported and (read_text or (not board and not reveal)):
            result.scores, result.limit = self._read_scores(im)
        return result

    def _stones(self, im, hsv):
        mask = cv2.inRange(hsv, np.array([12, 15, 195]), np.array([40, 230, 255]))
        mask[560:] = cv2.inRange(hsv[560:], np.array([12, 15, 75]), np.array([40, 230, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (18 < min(w, h) < 85 and 1.65 < max(w, h) / min(w, h) < 2.3):
                continue
            if cv2.contourArea(contour) / (w * h) < 0.85:
                continue
            crop = im[y + 3 : y + h - 3, x + 3 : x + w - 3]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark = (gray < gray.max() * 0.52).astype("uint8") * 255
            _, _, stats, centers = cv2.connectedComponentsWithStats(dark)
            counts = [0, 0]
            for stat, center in zip(stats[1:], centers[1:]):
                a, b, ww, hh, area = stat
                if not max(3, min(w, h) ** 2 * 0.012) <= area <= min(w, h) ** 2 * 0.15:
                    continue
                if not 0.5 <= ww / hh <= 2:
                    continue
                if a == 0 or b == 0 or a + ww == crop.shape[1] or b + hh == crop.shape[0]:
                    continue
                counts[int(center[0 if w > h else 1] > crop.shape[1 if w > h else 0] / 2)] += 1
            if max(counts) <= 6:
                found.append(StoneObservation(tuple(counts), (x, y, w, h)))
        return found

    def _text(self, crop):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        rows, _ = self._ocr(crop)
        return " ".join(row[1] for row in (rows or []) if row[2] > 0.8)

    def _read_names(self, im):
        names = []
        for x, y, x2, y2 in [
            (100, 620, 250, 665),
            (230, 42, 410, 94),
            (765, 42, 958, 94),
            (1285, 42, 1500, 94),
        ]:
            names.append(self._text(im[y:y2, x:x2]).strip())
        return names if all(names) and len(set(names)) == 4 else None

    def _read_scores(self, im):
        import re

        values = []
        for x, y, x2, y2 in [(740, 88, 902, 135), (230, 88, 383, 135)]:
            text = self._text(cv2.resize(im[y:y2, x:x2], None, fx=3, fy=3))
            match = re.search(r"(\d+)\s*/\s*(50|101)", text)
            if not match:
                return None, None
            values.append(tuple(map(int, match.groups())))
        if values[0][1] != values[1][1]:
            return None, None
        return (values[0][0], values[1][0]), values[0][1]

    def _read_counts(self, im, own_count):
        import re

        counts = [own_count if own_count <= 7 else None]
        for x, y, x2, y2 in [(180, 99, 237, 162), (712, 99, 770, 164), (1245, 99, 1299, 164)]:
            text = self._text(cv2.resize(im[y:y2, x:x2], None, fx=3, fy=3))
            match = re.fullmatch(r"[0-7]", text.strip())
            counts.append(int(match[0]) if match else None)
        return tuple(counts)
