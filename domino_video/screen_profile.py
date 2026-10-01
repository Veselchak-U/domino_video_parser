"""The same field areas are used for OCR and diagnostic samples."""

import cv2
import numpy as np


class ScreenProfile:
    areas = {
        "name": [
            (100, 620, 250, 665),
            (230, 42, 410, 94),
            (765, 42, 958, 94),
            (1285, 42, 1500, 94),
        ],
        "count": [(180, 99, 237, 162), (712, 99, 770, 164), (1245, 99, 1299, 164)],
        "score": [(740, 88, 902, 135), (230, 88, 383, 135)],
    }
    positions = ["снизу", "слева сверху", "по центру сверху", "справа сверху"]
    result_names = [
        (350, 296, 516, 341),
        (900, 296, 1070, 341),
        (590, 296, 776, 341),
        (1157, 296, 1340, 341),
    ]

    def result_table(self, hsv: np.ndarray) -> bool:
        green = cv2.inRange(hsv, np.array([35, 90, 70]), np.array([100, 255, 255]))
        # Four separated green cards, unlike the continuous green playing table.
        cards = [(350, 500), (600, 760), (910, 1040), (1190, 1300)]
        gaps = [(560, 565), (833, 844), (1119, 1128)]
        return all(
            cv2.countNonZero(green[365:410, a:b]) > (b - a) * 45 * 0.7 for a, b in cards
        ) and all(cv2.countNonZero(green[365:410, a:b]) < (b - a) * 45 * 0.3 for a, b in gaps)

    def region(self, field, seat=None, team=None):
        index = (seat - (2 if field == "count" else 1)) if seat else (0 if team == "A" else 1)
        return self.areas[field][index]

    def crop(self, normalized, field, seat=None, team=None):
        x, y, x2, y2 = self.region(field, seat, team)
        crop = normalized[y:y2, x:x2].copy()
        return crop if field == "name" else cv2.resize(crop, None, fx=3, fy=3)
