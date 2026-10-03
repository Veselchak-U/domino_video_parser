"""Визуальный профиль записи: координаты нормализованы к 1608×720."""

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .avatar_timer import AvatarTimerReader
from .name_ocr import NameOCR, NameReading
from .ocr_result import NAME_OCR_THRESHOLD, OCR_THRESHOLD, OCRResult
from .recognition_plan import OCRFields
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
    result_table: bool = False
    selective: bool = False
    reveal_points: tuple[int | None, ...] | None = None
    reveal_panels_complete: tuple[bool, ...] | None = None
    active_method: str | None = None
    timer_value: int | None = None
    timer_status: str | None = None


@dataclass
class PreparedObservation:
    observation: Observation
    crops: dict[str, list[np.ndarray | None]]

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
                region=(
                    ScreenProfile.result_names[seat - 1]
                    if field == "name" and result.result_table
                    else ScreenProfile().region(field, seat, team)
                ),
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
        if "counts" in self.crops:
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
                match = re.search(r"(\d+)\s*(?:\(\s*\+\s*\d+\s*\))?\s*/\s*(50|101)(?!\d)", value)
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
        if "reveal_points" in self.crops:
            points = []
            for i, crop in enumerate(self.crops["reveal_points"]):
                if crop is None:
                    points.append(None)
                    continue
                value, attempt = text(crop, "reveal_points", seat=i + 1)
                match = re.fullmatch(r"\d{1,2}", value)
                number = int(match[0]) if match else None
                if number is None or number > 84:
                    number = None
                    if attempt["reason"] is None:
                        attempt["reason"] = "invalid_format"
                points.append(number)
            result.reveal_points = tuple(points)
        return result


class ScreenRecognizer:
    def __init__(self):
        self._ocr = None
        self._name_ocr = None
        self._timer = AvatarTimerReader()

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
            score = cv2.countNonZero(blue)
            if ScreenProfile().result_table(hsv):
                score += 100000
            candidates.append((score, im))
        return max(candidates, key=lambda pair: pair[0])[1]

    def observe(self, image, time, read_text: bool | OCRFields = False):
        return self.prepare(image, time, read_text).finish(self._read, self._read_name)

    def prepare(self, image, time, read_text: bool | OCRFields = False, read_motion=False):
        im = self.normalize(image)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        profile = ScreenProfile()
        result_table = profile.result_table(hsv)
        table = hsv[150:550, 200:1450]
        green = cv2.inRange(table, np.array([35, 100, 20]), np.array([100, 255, 255]))
        supported = cv2.countNonZero(green) > 150000 and not result_table
        level = float(np.median(table[:, :, 2][green > 0])) if supported else 0
        reveal = supported and level < 80
        tiles = self._stones(im, hsv, read_motion)
        board = []
        uncertain_board = []
        hands = [[], [], [], []]
        uncertain_hands = [False] * 4
        # Native recovery also follows incoming tiles above the table, before
        # they disappear under the upper avatars. Coarse board sampling does not.
        board_top = 75 if read_motion else 140
        for tile in tiles:
            x, y, w, h = tile.box
            if tile.stone is None:
                if reveal:
                    if 45 < y < 150 and 100 < x < 1450:
                        uncertain_hands[1 if x < 500 else (2 if x < 1050 else 3)] = True
                    elif 390 < y < 470 and 500 < x < 1150:
                        uncertain_hands[0] = True
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
            elif board_top < y < 550 and 180 < x < 1480:
                board.append(tile)
        active, active_method = profile.active_indicator(hsv) if not reveal else (None, "none")
        timer = self._timer.read(im) if supported and not reveal and not read_motion else None
        if timer is not None and timer.status == "confirmed":
            active, active_method = timer.seat, "timer"
        elif timer is not None and timer.status == "ambiguous":
            active, active_method = None, "ambiguous_timer"
        elif timer is not None:
            # A remaining rim after the digits disappear is not a fresh turn.
            active, active_method = None, "timer_absent"
        result = Observation(time, board, hands, active, reveal, supported)
        result.active_method = active_method
        if timer is not None:
            result.timer_value, result.timer_status = timer.value, timer.status
        result.result_table = result_table
        result.selective = isinstance(read_text, OCRFields)
        result.uncertain_board = uncertain_board
        result.hand_regions_valid = (supported and not reveal, False, False, False)
        if reveal:
            result.reveal_panels_complete = tuple(
                profile.reveal_panel_complete(hsv, seat) for seat in range(1, 5)
            )
            # Reveal settles progressively. A darkened, unobstructed area is
            # necessary; zero detected tiles alone does not prove an empty hand.
            result.reveal_valid = tuple(
                supported
                and not uncertain_hands[seat]
                and float(np.median(hsv[y1:y2, x1:x2, 2])) < 110
                for seat, (x1, y1, x2, y2) in enumerate(
                    [
                        (500, 385, 1150, 480),
                        (100, 45, 500, 150),
                        (500, 45, 1050, 150),
                        (1050, 45, 1450, 150),
                    ]
                )
            )
        crops = {}
        fields = read_text if isinstance(read_text, OCRFields) else None
        names = fields.names if fields else bool(read_text)
        counts = fields.counts if fields else bool(read_text)
        scores = fields.scores if fields else bool(read_text) or (not board and not reveal)
        if names and result_table:
            crops["names"] = [im[y:y2, x:x2].copy() for x, y, x2, y2 in profile.result_names]
        elif names and fields is None and supported and not reveal:
            crops["names"] = [profile.crop(im, "name", seat=i) for i in range(1, 5)]
        if counts and supported and not reveal:
            crops["counts"] = [profile.crop(im, "count", seat=i) for i in range(2, 5)]
        if scores and supported:
            crops["scores"] = [profile.crop(im, "score", team=team) for team in "AB"]
        if reveal and (fields.reveal_points if fields else bool(read_text)):
            crops["reveal_points"] = [
                profile.crop(im, "reveal_points", seat=seat)
                if result.reveal_panels_complete[seat - 1]
                else None
                for seat in range(1, 5)
            ]
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
            # The lower avatar's flag protrudes beyond its circular portrait.
            # Its top edge can hide a pip while leaving a rectangular tile contour.
            overlaps_flag = x + 3 < 257 and x + w - 3 > 216 and y + 3 < 522 and y + h - 3 > 488
            if overlaps_avatar and 140 < y < 550 and 180 < x < 1480:
                found.append(StoneObservation((None, None), (x, y, w, h)))
                continue
            cw, ch = crop.shape[1] + 6, crop.shape[0] + 6
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark = (gray < gray.max() * 0.52).astype("uint8") * 255
            _, _, stats, centers = cv2.connectedComponentsWithStats(dark)
            counts = [0, 0]
            pip_centers = [[], []]
            for stat, center in zip(stats[1:], centers[1:]):
                a, b, ww, hh, area = stat
                if not max(3, min(cw, ch) ** 2 * 0.012) <= area <= min(cw, ch) ** 2 * 0.15:
                    continue
                if not 0.5 <= ww / hh <= 2:
                    continue
                if a == 0 or b == 0 or a + ww == crop.shape[1] or b + hh == crop.shape[0]:
                    continue
                half = int(center[0 if cw > ch else 1] > crop.shape[1 if cw > ch else 0] / 2)
                counts[half] += 1
                pip_centers[half].append((center[0] + x + 3, center[1] + y + 3))
            if max(counts) <= 6:
                values = tuple(counts)
                if overlaps_flag and not self._flag_reading_complete((x, y, w, h), pip_centers):
                    values = (None, None)
                found.append(StoneObservation(values, (x, y, w, h)))
        return found

    def _flag_reading_complete(self, box, centers):
        # Templates only reject incomplete readings; never fill in hidden pips.
        low, middle, high = 0.23, 0.5, 0.77
        corners = [(a, b) for a in (low, high) for b in (low, high)]
        diagonals = [[(low, low), (high, high)], [(low, high), (high, low)]]
        patterns = [[], [(middle, middle)], *diagonals]
        patterns += [points + [(middle, middle)] for points in diagonals]
        patterns += [corners, corners + [(middle, middle)]]
        patterns += [
            [(a, b) for a in (low, high) for b in (low, middle, high)],
            [(a, b) for a in (low, middle, high) for b in (low, high)],
        ]
        x, y, w, h = box
        width, height = (w / 2, h) if w > h else (w, h / 2)
        radius = min(width, height) * 0.12
        tolerance = min(width, height) * 0.18
        for half, observed in enumerate(centers):
            left = x + (half * width if w > h else 0)
            top = y + (half * height if h > w else 0)
            possible = set()
            for pattern in patterns:
                points = [(left + a * width, top + b * height) for a, b in pattern]
                unmatched = list(points)
                for cx, cy in observed:
                    nearest = min(
                        unmatched, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2, default=None
                    )
                    if (
                        nearest is None
                        or (nearest[0] - cx) ** 2 + (nearest[1] - cy) ** 2 > tolerance**2
                    ):
                        break
                    unmatched.remove(nearest)
                else:
                    if all(
                        216 - radius < px < 257 + radius and 488 - radius < py < 522 + radius
                        for px, py in unmatched
                    ):
                        possible.add(len(points))
            if possible != {len(observed)}:
                return False
        return True

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
        self._timer = AvatarTimerReader()
