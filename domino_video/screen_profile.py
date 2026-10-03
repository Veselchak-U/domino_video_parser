"""The same field areas are used for OCR and diagnostic samples."""

import cv2
import numpy as np


class ScreenProfile:
    avatar_centers = [(188, 538), (157, 80), (691, 80), (1224, 80)]
    areas = {
        "name": [
            (100, 620, 250, 665),
            (230, 42, 410, 94),
            (765, 42, 958, 94),
            (1285, 42, 1500, 94),
        ],
        "count": [(180, 99, 237, 162), (712, 99, 770, 164), (1245, 99, 1299, 164)],
        "score": [(740, 88, 902, 135), (230, 88, 383, 135)],
        "reveal_points": [
            (756, 564, 808, 616),
            (165, 208, 217, 260),
            (699, 208, 751, 260),
            (1235, 208, 1287, 260),
        ],
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

    def active_player(self, hsv: np.ndarray) -> int | None:
        return self.active_indicator(hsv)[0]

    def active_indicator(self, hsv: np.ndarray) -> tuple[int | None, str]:
        yy, xx = np.ogrid[-58:59, -58:59]
        radius = xx * xx + yy * yy
        ring = (48**2 <= radius) & (radius < 58**2)
        sectors = ((np.arctan2(yy, xx) + np.pi) * (4 / np.pi)).astype(int) % 8
        cells = [ring & (sectors == sector) for sector in range(8)]
        candidates, intensity = [], []
        for seat, (x, y) in enumerate(self.avatar_centers):
            orange = cv2.inRange(
                hsv[y - 58 : y + 59, x - 58 : x + 59],
                np.array([8, 120, 170]),
                np.array([30, 255, 255]),
            )
            intensity.append(cv2.countNonZero(orange[13:104, 13:104]))
            occupied = [float(np.mean(orange[cell] > 0)) >= 0.25 for cell in cells]
            # An extended rim distinguishes the timer from a portrait or a
            # localized medal; calibrated on 17 real frames, not all decorations.
            if any(
                all(occupied[(start + offset) % 8] for offset in range(3)) for start in range(8)
            ):
                candidates.append(seat)
        if candidates:
            return (candidates[0], "arc") if len(candidates) == 1 else (None, "ambiguous_arc")
        # Ornate frames can hide the rim while leaving the old timer signal.
        return (
            (int(np.argmax(intensity)), "avatar_color") if max(intensity) > 500 else (None, "none")
        )

    def region(self, field, seat=None, team=None):
        index = (seat - (2 if field == "count" else 1)) if seat else (0 if team == "A" else 1)
        return self.areas[field][index]

    def reveal_panel_complete(self, hsv, seat):
        x, y, x2, y2 = self.region("reveal_points", seat=seat)
        cx, cy = (x + x2) // 2, (y + y2) // 2
        values = hsv[cy - 34 : cy + 35, cx - 34 : cx + 35, 2]
        yy, xx = np.ogrid[: values.shape[0], : values.shape[1]]
        radius = (xx - 34) ** 2 + (yy - 34) ** 2
        ring = (26**2 <= radius) & (radius <= 30**2)
        # The result badge expands from a dot. Only its full bright rim proves
        # this player's reveal animation has reached the points panel.
        return float(np.mean(values[ring] >= 100)) >= 0.9

    def crop(self, normalized, field, seat=None, team=None):
        x, y, x2, y2 = self.region(field, seat, team)
        crop = normalized[y:y2, x:x2].copy()
        return crop if field == "name" else cv2.resize(crop, None, fx=3, fy=3)
