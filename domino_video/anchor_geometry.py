"""Direct two-anchor identity evidence, without changing events or pip values."""

import math
from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vision import Observation, StoneObservation


def _point(tile: "StoneObservation") -> complex:
    return complex(*tile.center)


def _scaled_dimensions(old, new, scale):
    # Initial calibration: 16:15 gives ~0.73 scale for both anchors. A 15%
    # allowance rejects incompatible pairs without relaxing the centre gate.
    return (
        scale > 0
        and min(*old.box[2:], *new.box[2:]) > 0
        and all(
            0.85 <= b / (scale * a) <= 1.15
            for a, b in zip(sorted(old.box[2:]), sorted(new.box[2:]))
        )
    )


def anchor_matches(
    hidden: "StoneObservation",
    anchors: list["StoneObservation"],
    early_board: list["StoneObservation"],
    late_board: list["StoneObservation"],
) -> list[dict]:
    """Return all matching values/pairs; callers must reject competing matches.

    Labels identify the anchors, never the hidden value. Complex multiplication
    permits translation, positive scale and rotation, but never reflection.
    """
    old_counts = Counter(tile.stone for tile in early_board)
    new_counts = Counter(tile.stone for tile in late_board)
    visible = {tile.stone: tile for tile in late_board}
    matches = []
    for pair in combinations(anchors, 2):
        keys = [tile.stone for tile in pair]
        if keys[0] == keys[1] or any(
            key is None or old_counts[key] != 1 or new_counts[key] != 1 for key in keys
        ):
            continue
        a, b = pair
        first, second = [visible[key] for key in keys]
        before, after = _point(b) - _point(a), _point(second) - _point(first)
        # A base shorter than a tile poorly constrains scale/rotation. The
        # provisional 0.75 bound is below the real adjacent-anchor separation.
        if abs(before) < 0.75 * max(*a.box[2:], *b.box[2:]) or abs(after) < 0.75 * max(
            *first.box[2:], *second.box[2:]
        ):
            continue
        z = after / before
        scale = abs(z)
        if not all(_scaled_dimensions(old, new, scale) for old, new in zip(pair, (first, second))):
            continue
        if any(abs(_point(hidden) - _point(tile)) > 1.75 * max(tile.box[2:]) for tile in pair):
            continue
        prediction = _point(first) + (_point(hidden) - _point(a)) * z
        for tile in late_board:
            if tile.stone is None or tile.stone in keys or new_counts[tile.stone] != 1:
                continue
            error = abs(_point(tile) - prediction)
            if error >= 0.6 * min(tile.box[2:]) or not _scaled_dimensions(hidden, tile, scale):
                continue
            if any(
                abs(_point(tile) - _point(anchor)) > 1.75 * max(anchor.box[2:])
                for anchor in (first, second)
            ):
                continue
            matches.append(
                dict(
                    stone=tile.stone,
                    anchors=keys,
                    early_centers=[list(t.center) for t in pair],
                    late_centers=[list(t.center) for t in (first, second)],
                    scale=scale,
                    angle=math.atan2(z.imag, z.real),
                    error=error,
                    predicted_center=[prediction.real, prediction.imag],
                )
            )
    return matches


def repeated_identity(
    hidden: "StoneObservation",
    anchors: list["StoneObservation"],
    early_board: list["StoneObservation"],
    observations: list["Observation"],
) -> dict | None:
    """Require exactly one value supported by three distinct late PTS."""
    proofs = []
    readings = {}
    geometries = {}
    for obs in observations:
        # Native video repeats the same layout for hundreds of PTS. Reuse only
        # exact geometry; every timestamp still contributes separately below.
        geometry = tuple((tuple(tile.values), tuple(tile.box)) for tile in obs.board)
        if geometry not in geometries:
            geometries[geometry] = anchor_matches(hidden, anchors, early_board, obs.board)
        matches = geometries[geometry]
        for match in matches:
            readings.setdefault(match["stone"], set()).add(obs.time)
        proofs.extend(dict(time=obs.time, **match) for match in matches)
    values = [stone for stone, times in readings.items() if len(times) >= 3]
    if len(values) != 1:
        return None
    stone = values[0]
    return dict(
        stone=stone,
        readings=[proof for proof in proofs if proof["stone"] == stone],
        unconfirmed_values={key: sorted(times) for key, times in readings.items() if key != stone},
    )


def static_slot_bridge(
    first: "StoneObservation",
    last: "StoneObservation",
    local_anchors: list["StoneObservation"],
    observations: list["Observation"],
) -> dict | None:
    """Prove one stationary place across an observed occlusion, never a flight."""
    if len(observations) < 2 or abs(_point(first) - _point(last)) >= 4:
        return None
    if not _scaled_dimensions(first, last, 1):
        return None
    if any(not 0 < b.time - a.time <= 0.15 for a, b in zip(observations, observations[1:])):
        return None
    maps = []
    for obs in observations:
        counts = Counter(tile.stone for tile in obs.board)
        maps.append(
            {
                tile.stone: tile
                for tile in obs.board
                if tile.stone is not None and counts[tile.stone] == 1
            }
        )
        if any(abs(_point(tile) - _point(first)) < 0.6 * min(first.box[2:]) for tile in obs.board):
            return None
        if (
            sum(
                abs(_point(tile) - _point(first)) < 0.6 * min(first.box[2:])
                for tile in obs.uncertain_board
            )
            > 1
        ):
            return None
    shared = set.intersection(*(set(mapping) for mapping in maps))
    fixed = {
        key
        for key in shared
        if all(
            abs(_point(mapping[key]) - _point(maps[0][key])) < 4
            and _scaled_dimensions(maps[0][key], mapping[key], 1)
            for mapping in maps
        )
    }
    local = fixed & {tile.stone for tile in local_anchors}
    pairs = [
        (key, other)
        for key in local
        for other in fixed - {key}
        if 0.75 * max(maps[0][key].box[2:])
        <= abs(_point(maps[0][key]) - _point(maps[0][other]))
        <= 1.75 * max(maps[0][key].box[2:])
    ]
    if not pairs:
        return None
    return dict(
        times=[o.time for o in observations], anchors=sorted({k for pair in pairs for k in pair})
    )
