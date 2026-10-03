"""Physical identity through an occluded installation; no OCR or game mutation."""

import math


def _segments(rows, stone, include_unknown=False):
    segments = []
    for obs in rows:
        for tile in obs.board + obs.uncertain_board:
            # Native frames may retain the physical contour while OCR is
            # hidden by an avatar or counter. Keep unknown contours in the
            # track; identity is established later by the settled neighbour.
            if tile.stone != stone and not (include_unknown and tile.stone is None):
                continue
            nearby = [
                s
                for s in segments
                if 0 < obs.time - s[-1][0].time <= 0.15
                and math.dist(tile.center, s[-1][1].center)
                < max(*tile.box[2:], *s[-1][1].box[2:])
                * 1.5
                * max(1, (obs.time - s[-1][0].time) / 0.02)
            ]
            if len(nearby) == 1:
                nearby[0].append((obs, tile))
            elif not nearby:
                segments.append([(obs, tile)])
    return segments


def _extend(track, rows, stone, stop, allow_missing=False):
    result = list(track)
    for obs in rows:
        if obs.time <= track[-1][0].time or obs.time > stop:
            continue
        before, tile = result[-1]
        if obs.time - before.time > 0.15:
            break
        candidates = [
            t
            for t in obs.board + obs.uncertain_board
            if math.dist(t.center, tile.center) < min(tile.box[2:]) * 0.8
        ]
        permitted = [t for t in candidates if t.stone in (None, stone)]
        conflicts = [
            t
            for t in candidates
            if t.stone not in (None, stone)
            and math.dist(t.center, tile.center) < min(tile.box[2:]) * 0.4
        ]
        if len(permitted) > 1 or conflicts:
            return None
        if not permitted:
            if allow_missing:
                continue
            break
        result.append((obs, permitted[0]))
    return result


def occluded_placements(rows, events):
    """Return every admissible track; the caller enforces uniqueness/old identity."""
    results = {}
    for event in events:
        stone = event["stone"]
        if event.get("recovery", {}).get("evidence", {}).get("motion") == "incoming":
            continue
        simultaneous = sum(other["time"] == event["time"] for other in events) > 1
        for segment in _segments(rows, stone, include_unknown=simultaneous):
            if len({o.time for o, _ in segment}) < 5 or segment[-1][0].time >= event["time"]:
                continue
            track = _extend(segment, rows, stone, event["time"])
            if not track or not any(t.stone is None for _, t in track):
                continue
            tail = track[-3:]
            if len(tail) < 3 or any(
                math.dist(a[1].center, b[1].center) >= 4 for a, b in zip(tail, tail[1:])
            ):
                continue
            neighbours = []
            for old in events:
                if old["stone"] == stone or old["time"] >= track[0][0].time:
                    continue
                starts = [
                    (o, t)
                    for o in rows
                    if old["time"] <= o.time < track[0][0].time
                    for t in o.board
                    if t.stone == old["stone"]
                ]
                if not starts:
                    continue
                # The last visible old identity seeds a unique physical track.
                anchor = _extend(starts[-1:], rows, old["stone"], track[-1][0].time, True)
                if not anchor or sum(t.stone == old["stone"] for _, t in anchor) < 3:
                    continue
                by_time = {o.time: t for o, t in anchor}
                if any(o.time not in by_time for o, _ in tail):
                    continue
                settled = [by_time[o.time] for o, _ in tail]
                if any(math.dist(a.center, b.center) >= 4 for a, b in zip(settled, settled[1:])):
                    continue
                # The settled contour must be directly adjacent to the old
                # neighbour; a distant parallel chain (such as the upper
                # avatar stack) is not an identity proof.
                if any(
                    math.dist(t.center, by_time[o.time].center) >= max(t.box[2:]) * 1.5
                    for o, t in tail
                ):
                    continue
                expected = event.get("positions", {}).get(stone)
                old_expected = event.get("positions", {}).get(old["stone"])
                if expected and old_expected:
                    if math.dist(expected, old_expected) >= event.get("tile_size", 100) * 1.3:
                        continue
                distances = [
                    (o.time, math.dist(t.center, by_time[o.time].center))
                    for o, t in track
                    if o.time in by_time
                ]
                if len(distances) < 5 or distances[0][0] != track[0][0].time:
                    continue
                if distances[0][1] - distances[-1][1] <= min(tail[-1][1].box[2:]) * 0.5:
                    continue
                if any(b[1] > a[1] + 4 for a, b in zip(distances, distances[1:])):
                    continue
                neighbours.append(
                    dict(neighbor=old["stone"], neighbor_track=anchor, distances=distances)
                )
            if len(neighbours) == 1:
                results.setdefault(stone, []).append(dict(track=track, **neighbours[0]))
            elif not neighbours and event.get("positions"):
                # When the last few native frames hide both tiles under the
                # avatar, retain the uniquely adjacent old end if the target
                # contour itself proves an incoming flight.  The old stone is
                # still required to be independently known and the endpoint
                # must be the unique nearest end in the final layout.
                expected = event["positions"].get(stone)
                choices = []
                for old in events:
                    if old["stone"] == stone or old["time"] >= track[0][0].time:
                        continue
                    old_expected = event["positions"].get(old["stone"])
                    if not expected or not old_expected:
                        continue
                    if math.dist(expected, old_expected) >= event.get("tile_size", 100) * 1.3:
                        continue
                    distances = [
                        (o.time, math.dist(tile.center, old_expected)) for o, tile in track
                    ]
                    if len(distances) < 5:
                        continue
                    if distances[-1][1] >= event.get("tile_size", 100) * 1.6:
                        continue
                    if (
                        math.dist(track[0][1].center, track[-1][1].center)
                        <= event.get("tile_size", 100) * 0.5
                    ):
                        continue
                    choices.append((old["stone"], distances))
                if len(choices) == 1:
                    neighbor, distances = choices[0]
                    results.setdefault(stone, []).append(
                        dict(track=track, neighbor=neighbor, neighbor_track=[], distances=distances)
                    )
    return results
