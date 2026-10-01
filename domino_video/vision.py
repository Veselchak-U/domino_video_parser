"""Визуальный профиль записи: координаты нормализованы к 1608×720."""

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .name_ocr import NameOCR, NameReading
from .ocr_result import NAME_OCR_THRESHOLD, OCR_THRESHOLD, OCRResult
from .screen_profile import ScreenProfile


@dataclass(frozen=True)
class StoneObservation:
    values: tuple[int | None, int | None]
    box: tuple[int, int, int, int]

    @property
    def stone(self):
        if None in self.values:
            return None
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
    ocr_attempts: list[dict] = field(default_factory=list)
    uncertain_board: list[StoneObservation] = field(default_factory=list)
    hand_regions_valid: tuple[bool, ...] | None = None
    reveal_valid: tuple[bool, ...] | None = None
    dense: bool = False


@dataclass
class PreparedObservation:
    observation: Observation
    crops: dict[str, list[np.ndarray]]

    def finish(self, read, read_name=None):
        result = self.observation
        result.ocr_attempts = []

        def text(crop, field, seat=None, team=None):
            reading = (read_name or read)(crop) if field == "name" else read(crop)
            accepted = reading.text
            if field != "score" and not isinstance(reading, NameReading):
                accepted = accepted.strip()
            attempt = dict(
                field=field,
                seat=seat,
                team=team,
                time=result.time,
                raw_rows=[asdict(row) for row in reading.rows],
                accepted_text=accepted,
                threshold=NAME_OCR_THRESHOLD if field == "name" else OCR_THRESHOLD,
                reason=reading.reason,
                region=ScreenProfile().region(field, seat, team),
                scale=1 if field == "name" else 3,
            )
            result.ocr_attempts.append(attempt)
            if isinstance(reading, NameReading):
                attempt["symbols"] = [asdict(s) for s in reading.symbols]
                attempt["alternatives"] = list(reading.alternatives)
            return accepted, attempt

        if "names" in self.crops:
            readings = [
                text(crop, "name", seat=i + 1) for i, crop in enumerate(self.crops["names"])
            ]
            names = [value for value, _ in readings]
            duplicates = Counter(names)
            for value, attempt in readings:
                if value and duplicates[value] > 1:
                    attempt["reason"] = "duplicate_name"
            result.names = names if all(names) and len(set(names)) == 4 else None
            own_count = len(result.hands[0])
            counts = [own_count if own_count <= 7 else None]
            for i, crop in enumerate(self.crops["counts"]):
                value, attempt = text(crop, "count", seat=i + 2)
                match = re.fullmatch(r"[0-7]", value)
                if not match and attempt["reason"] is None:
                    attempt["reason"] = "invalid_format"
                counts.append(int(match[0]) if match else None)
            result.counts = tuple(counts)
        if "scores" in self.crops:
            values = []
            for i, crop in enumerate(self.crops["scores"]):
                value, attempt = text(crop, "score", team="AB"[i])
                match = re.search(r"(\d+)\s*/\s*(50|101)", value)
                if not match:
                    if attempt["reason"] is None:
                        attempt["reason"] = "invalid_format"
                    values.append(None)
                else:
                    values.append(tuple(map(int, match.groups())))
            if len(values) == 2 and all(values) and values[0][1] == values[1][1]:
                result.scores = (values[0][0], values[1][0])
                result.limit = values[0][1]
            elif len(values) == 2 and all(values):
                for attempt in result.ocr_attempts:
                    if attempt["field"] == "score":
                        attempt["reason"] = "invalid_format"
        return result


class ScreenRecognizer:
    def __init__(self):
        self._ocr = None
        self._name_ocr = None

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
        return self.prepare(image, time, read_text).finish(self._read, self._read_name)

    def prepare(self, image, time, read_text=False, read_motion=False):
        im = self.normalize(image)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        table = hsv[150:550, 200:1450]
        green = cv2.inRange(table, np.array([35, 100, 20]), np.array([100, 255, 255]))
        supported = cv2.countNonZero(green) > 150000
        level = float(np.median(table[:, :, 2][green > 0])) if supported else 0
        reveal = supported and level < 80
        tiles = self._stones(im, hsv, read_motion)
        board = []
        uncertain_board = []
        hands = [[], [], [], []]
        for tile in tiles:
            x, y, w, h = tile.box
            if tile.stone is None:
                if not reveal and 140 < y < 550 and 180 < x < 1480:
                    uncertain_board.append(tile)
                continue
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
        result.uncertain_board = uncertain_board
        result.hand_regions_valid = (supported and not reveal, False, False, False)
        if reveal:
            # Reveal settles progressively. A darkened, unobstructed area is
            # necessary; zero detected tiles alone does not prove an empty hand.
            result.reveal_valid = tuple(
                supported and float(np.median(hsv[y1:y2, x1:x2, 2])) < 110
                for x1, y1, x2, y2 in [
                    (500, 385, 1150, 480),
                    (100, 45, 500, 150),
                    (500, 45, 1050, 150),
                    (1050, 45, 1450, 150),
                ]
            )
        crops = {}
        profile = ScreenProfile()
        if read_text and supported and not reveal:
            crops["names"] = [profile.crop(im, "name", seat=i) for i in range(1, 5)]
            crops["counts"] = [profile.crop(im, "count", seat=i) for i in range(2, 5)]
        if supported and (read_text or (not board and not reveal)):
            crops["scores"] = [profile.crop(im, "score", team=team) for team in "AB"]
        return PreparedObservation(result, crops)

    def _stones(self, im, hsv, read_motion=False):
        mask = cv2.inRange(hsv, np.array([12, 15, 195]), np.array([40, 230, 255]))
        mask[560:] = cv2.inRange(hsv[560:], np.array([12, 15, 75]), np.array([40, 230, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (18 < min(w, h) < 85 and 1.65 < max(w, h) / min(w, h) < 2.3):
                if not read_motion:
                    continue
                rect = cv2.minAreaRect(contour)
                short, long = sorted(rect[1])
                if not (18 < short < 85 and 1.65 < long / short < 2.3):
                    continue
                if cv2.contourArea(contour) / (short * long) < 0.85:
                    continue
                points = cv2.boxPoints(rect)
                center = points.mean(axis=0)
                points = points[
                    np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
                ]
                points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
                cw = int(np.linalg.norm(points[1] - points[0]))
                ch = int(np.linalg.norm(points[2] - points[1]))
                target = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
                crop = cv2.warpPerspective(
                    im, cv2.getPerspectiveTransform(np.float32(points), target), (cw, ch)
                )
                crop = crop[3:-3, 3:-3]
            elif cv2.contourArea(contour) / (w * h) < 0.85:
                if cv2.contourArea(contour) / (w * h) >= 0.45:
                    found.append(StoneObservation((None, None), (x, y, w, h)))
                continue
            else:
                crop = im[y + 3 : y + h - 3, x + 3 : x + w - 3]
            overlaps_avatar = any(
                (max(x, min(ax, x + w)) - ax) ** 2 + (max(y, min(ay, y + h)) - ay) ** 2 < 45**2
                for ax, ay in [(188, 538), (157, 80), (691, 80), (1224, 80)]
            )
            if overlaps_avatar and 140 < y < 550 and 180 < x < 1480:
                found.append(StoneObservation((None, None), (x, y, w, h)))
                continue
            cw, ch = crop.shape[1] + 6, crop.shape[0] + 6
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark = (gray < gray.max() * 0.52).astype("uint8") * 255
            _, _, stats, centers = cv2.connectedComponentsWithStats(dark)
            counts = [0, 0]
            for stat, center in zip(stats[1:], centers[1:]):
                a, b, ww, hh, area = stat
                if not max(3, min(cw, ch) ** 2 * 0.012) <= area <= min(cw, ch) ** 2 * 0.15:
                    continue
                if not 0.5 <= ww / hh <= 2:
                    continue
                if a == 0 or b == 0 or a + ww == crop.shape[1] or b + hh == crop.shape[0]:
                    continue
                counts[int(center[0 if cw > ch else 1] > crop.shape[1 if cw > ch else 0] / 2)] += 1
            if max(counts) <= 6:
                found.append(StoneObservation(tuple(counts), (x, y, w, h)))
        return found

    def _text(self, crop):
        return self._read(crop).text

    def _read(self, crop):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        rows, _ = self._ocr(crop)
        return OCRResult.from_rows(rows)

    def _read_name(self, crop):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        if self._name_ocr is None:
            self._name_ocr = NameOCR(self._ocr)
        return self._name_ocr.read(crop)

    def close(self):
        if self._name_ocr is not None:
            self._name_ocr.close()
        self._name_ocr = self._ocr = None
