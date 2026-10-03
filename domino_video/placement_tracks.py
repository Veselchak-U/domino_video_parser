"""Separate same-value contours without selecting an ambiguous physical identity."""

import math
from itertools import groupby


def placement_tracks(readings):
    tracks = []
    ambiguous = set()
    for time, group in groupby(sorted(readings, key=lambda r: r[0].time), key=lambda r: r[0].time):
        current = list({tuple(tile.box): (obs, tile) for obs, tile in group}.values())
        links = {}
        for index, (_, tile) in enumerate(current):
            candidates = []
            for number, track in enumerate(tracks):
                previous, old = track[-1]
                if not 0 < time - previous.time <= 0.15:
                    continue
                sizes = (max(old.box[2:]), max(tile.box[2:]))
                # Provisional physical scale/step bounds; placement itself is
                # still proved independently by the established incoming guard.
                if (
                    max(sizes) <= min(sizes) * 1.5
                    and math.dist(old.center, tile.center) <= max(sizes) * 2
                ):
                    candidates.append(number)
            links[index] = candidates
        for index, reading in enumerate(current):
            candidates = links[index]
            unique = (
                len(candidates) == 1
                and sum(candidates[0] in peers for peers in links.values()) == 1
            )
            if unique:
                tracks[candidates[0]].append(reading)
            else:
                if candidates:
                    ambiguous.update(candidates)
                    ambiguous.add(len(tracks))
                tracks.append([reading])
    return [track for number, track in enumerate(tracks) if number not in ambiguous]
