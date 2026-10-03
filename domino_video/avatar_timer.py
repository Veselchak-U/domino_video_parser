"""Read the digital countdown independently of decorative avatar rims."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .screen_profile import ScreenProfile


@dataclass(frozen=True)
class AvatarTimerReading:
    seat: int | None = None
    value: int | None = None
    confidence: float = 0.0
    status: Literal["absent", "confirmed", "ambiguous"] = "absent"


class AvatarTimerReader:
    """Lazy OCR after a cheap glyph gate, with 64 background-free mask samples.

    Coordinates and thresholds apply only to the existing normalized screen.
    This reader owns no turn history; a disappearance never selects another seat.
    """

    def __init__(self):
        self._engine = None
        self._samples: list[tuple[np.ndarray, int, float]] = []

    def read(self, normalized_bgr: np.ndarray) -> AvatarTimerReading:
        candidates = []
        for seat, (x, y) in enumerate(ScreenProfile.avatar_centers):
            crop = normalized_bgr[y - 25 : y + 25, x - 37 : x + 37]
            mask = self._glyph_mask(crop)
            if mask is not None:
                candidates.append((seat, crop, mask))
        if not candidates:
            return AvatarTimerReading()
        # Even an unreadable second set of glyphs prevents arbitrary selection.
        if len(candidates) != 1:
            return AvatarTimerReading(status="ambiguous")
        seat, crop, mask = candidates[0]
        for index, (sample, value, confidence) in enumerate(self._samples):
            union = np.count_nonzero(sample | mask)
            similarity = np.count_nonzero(sample & mask) / union
            if similarity >= 0.90:
                self._samples.append(self._samples.pop(index))
                return AvatarTimerReading(seat, value, confidence, "confirmed")
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        readings, _ = self._engine(cv2.resize(crop, None, fx=3, fy=3), use_det=False, use_cls=False)
        text, confidence = readings[0] if readings else ("", 0.0)
        if text.isascii() and text.isdigit() and 0 <= int(text) <= 30 and confidence >= 0.8:
            value = int(text)
            self._samples.append((mask, value, float(confidence)))
            self._samples = self._samples[-64:]
            return AvatarTimerReading(seat, value, float(confidence), "confirmed")
        return AvatarTimerReading(status="ambiguous")

    @staticmethod
    def _glyph_mask(crop: np.ndarray) -> np.ndarray | None:
        if crop.shape != (50, 74, 3):
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, np.array([8, 65, 100]), np.array([32, 255, 255]))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(orange)
        selected = []
        # Complete central glyphs exclude skin connected to the crop edge and
        # decorations. Include narrow '1' and single-digit countdowns as well.
        for label, (x, y, width, height, area) in enumerate(stats[1:], start=1):
            if 8 <= width <= 28 and 29 <= height <= 41 and 3 <= y <= 13 and area >= 110:
                if 9 <= x and x + width <= 66:
                    selected.append(label)
        if not 1 <= len(selected) <= 2:
            return None
        mask = np.isin(labels, selected).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(selected) == 1 and not 26 <= (xs.min() + xs.max()) / 2 <= 48:
            return None
        glyphs = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        return cv2.resize(glyphs, (48, 40), interpolation=cv2.INTER_NEAREST)
