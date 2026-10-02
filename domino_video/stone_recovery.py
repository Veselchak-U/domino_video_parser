"""Evidence for missed moves; no video I/O or publication of files."""

import math
from collections import Counter
from copy import deepcopy


class StoneRecovery:
    def augment(self, rounds, observations):
        for rnd in rounds:
            end = rnd["end"] if rnd["end"] is not None else float("inf")
            rows = [
                o
                for o in observations
                if rnd["start"] <= o.time < end and o.supported and not o.reveal
            ]
            rnd.setdefault("stone_recovery", [])
            rnd.setdefault("unresolved", [])
            self._hands(rnd, rows)
            self._late(rnd, rows)
            self._short(rnd, rows)
            rnd["events"].sort(key=lambda e: e["time"])
            for index, event in enumerate(rnd["events"], 1):
                event.setdefault("id", index)
                proof_time = event.get("recovery", {}).get("time")
                proof_time = proof_time if proof_time is not None else event["time"]
                visible = [
                    o
                    for o in rows
                    if proof_time <= o.time and any(s.stone == event["stone"] for s in o.board)
                ]
                if visible:
                    best = max(visible, key=lambda o: len(o.board))
                    event["positions"] = {s.stone: list(s.center) for s in best.board}
                    event["tile_size"] = max(
                        next(s.box[2:] for s in best.board if s.stone == event["stone"])
                    )
                if event.get("recovery"):
                    event["recovery"]["event_id"] = event["id"]
        return rounds

    def _entry(self, rnd, event, method, interval, time=None, box=None, evidence=None):
        for existing in rnd["stone_recovery"]:
            if (
                existing["method"] == method
                and existing["stone"] == event.get("stone")
                and existing["interval"] == list(interval)
            ):
                event["recovery"] = existing
                return existing
        entry = dict(
            event_id=event.get("id"),
            stone=event.get("stone"),
            seat=event["seat"] + 1
            if event.get("seat") is not None and len(event.get("seats", [event["seat"]])) == 1
            else None,
            interval=list(interval),
            method=method,
            time=time,
            region=list(box) if box else None,
            evidence=evidence or {},
            reason=None,
            sample=None,
            status="candidate",
        )
        rnd["stone_recovery"].append(entry)
        event["recovery"] = entry
        return entry

    def _hands(self, rnd, rows):
        stable = None
        candidate = None
        repeated = 0
        first = None
        for obs in rows:
            if obs.hand_regions_valid is not None and not obs.hand_regions_valid[0]:
                continue
            hand = tuple(sorted(obs.hands[0]))
            if not hand or len(hand) != len(set(hand)):
                candidate, repeated = None, 0
                continue
            if hand == candidate:
                repeated += 1
            else:
                candidate, repeated, first = hand, 1, obs.time
            if repeated < 2:
                continue
            if stable and stable[0] != hand:
                before, last = stable
                lost = set(before) - set(hand)
                if len(lost) == 1 and set(hand) < set(before) and first - last <= 3:
                    stone = next(iter(lost))
                    matches = [e for e in rnd["events"] if e["stone"] == stone]
                    if not matches:
                        event = dict(
                            time=first,
                            interval=[last, first],
                            seat=0,
                            seats=[0],
                            action=None,
                            stone=stone,
                        )
                        rnd["events"].append(event)
                        self._entry(
                            rnd,
                            event,
                            "hand_difference",
                            [last, first],
                            last,
                            [450, 560, 900, 160],
                            dict(
                                before=list(before), after=list(hand), times=[last, first, obs.time]
                            ),
                        )
                    elif all(0 not in e.get("seats", [e["seat"]]) for e in matches):
                        simultaneous = [
                            e for e in matches if last - 0.3 <= e["time"] <= first + 0.3
                        ]
                        if simultaneous:
                            for event in simultaneous:
                                event["seat"], event["seats"] = 0, [0]
                                self._entry(
                                    rnd,
                                    event,
                                    "hand_difference",
                                    [last, first],
                                    last,
                                    [450, 560, 900, 160],
                                    dict(
                                        before=list(before),
                                        after=list(hand),
                                        times=[last, first, obs.time],
                                        indicator_conflict=True,
                                    ),
                                )
                            rnd["indicator_unreliable"] = True
                stable = hand, obs.time
            else:
                stable = hand, obs.time
        if rnd.get("indicator_unreliable"):
            for event in rnd["events"]:
                if not event.get("recovery"):
                    event["seats"] = [0, 1, 2, 3]

    def _anchors(self, obs, tile):
        return sorted(
            (s for s in obs.board if tile.stone is None or s.stone != tile.stone),
            key=lambda s: math.dist(tile.center, s.center),
        )[:2]

    def _played_slot(self, slot, events, rows):
        if slot.get("ambiguous"):
            return False
        prior = [e for e in events if e["time"] < slot["interval"][0]]
        played = {e["stone"] for e in prior}
        layout_start = max((e["time"] for e in prior), default=slot["time"])
        hits = {}
        for obs in rows:
            # Older placements can rearrange bends in the chain, so their
            # geometry cannot identify a contour in the current layout.
            if obs.time < layout_start or slot["time"] <= obs.time <= slot["last"]:
                continue
            visible = {s.stone: s for s in obs.board}
            for stone in played & visible.keys():
                tile = visible[stone]
                compatible = []
                for old in slot["anchors"]:
                    anchor = visible.get(old.stone)
                    if anchor is None:
                        continue
                    scale = max(anchor.box[2:]) / max(old.box[2:])
                    predicted = tuple(
                        anchor.center[i] + (slot["tile"].center[i] - old.center[i]) * scale
                        for i in range(2)
                    )
                    compatible.append(math.dist(predicted, tile.center) < min(tile.box[2:]) * 0.6)
                if compatible and all(compatible):
                    hits.setdefault(stone, set()).add(obs.time)
        # A footprint alone cannot dismiss a missed placement: require one
        # previously played tile supported by repeated anchor-relative readings.
        return len([stone for stone, times in hits.items() if len(times) >= 3]) == 1

    def _late(self, rnd, rows):
        slots = []
        previous_time = rnd["start"]
        for obs in rows:
            counts = Counter(s.stone for s in obs.board)
            duplicated = [s for s in obs.board if counts[s.stone] > 1]
            for tile in obs.uncertain_board + duplicated:
                anchors = self._anchors(obs, tile)
                if not anchors:
                    continue
                existing = [
                    s
                    for s in slots
                    if math.dist(s["tile"].center, tile.center) < 4 and obs.time - s["last"] < 1
                ]
                if existing:
                    slot = existing[0]
                    if slot["last"] == obs.time:
                        slot["ambiguous"] = True
                    slot["count"] += 1
                    slot["last"] = obs.time
                else:
                    seats = sorted(
                        {
                            o.active
                            for o in rows
                            if previous_time - 3.5 <= o.time <= obs.time and o.active is not None
                        }
                    )
                    slots.append(
                        dict(
                            tile=tile,
                            anchors=anchors,
                            time=obs.time,
                            last=obs.time,
                            interval=[previous_time, obs.time],
                            count=1,
                            seats=seats,
                        )
                    )
            previous_time = obs.time
        slots = [slot for slot in slots if not self._played_slot(slot, rnd["events"], rows)]
        used = set()
        known = {e["stone"] for e in rnd["events"]}
        # A newly visible tile may overlap an old screen footprint after the
        # whole chain moves. Match it to an unknown slot before that footprint
        # filter can discard it as a duplicate/miscount.
        for slot in slots:
            if slot["count"] < 2 or slot.get("ambiguous"):
                continue
            candidates = {}
            for obs in rows:
                if obs.time <= slot["last"]:
                    continue
                visible = {s.stone: s for s in obs.board}
                for tile in obs.board:
                    if tile.stone in known:
                        continue
                    compatible = []
                    for old in slot["anchors"]:
                        anchor = visible.get(old.stone)
                        if anchor is not None:
                            scale = max(anchor.box[2:]) / max(old.box[2:])
                            predicted = tuple(
                                anchor.center[i] + (slot["tile"].center[i] - old.center[i]) * scale
                                for i in range(2)
                            )
                            compatible.append(
                                math.dist(predicted, tile.center) < min(tile.box[2:]) * 0.6
                            )
                    if compatible and all(compatible):
                        candidates.setdefault(tile.stone, []).append(obs.time)
            valid = [(stone, times) for stone, times in candidates.items() if len(times) >= 3]
            if len(valid) == 1:
                stone, times = valid[0]
                rnd["events"].append(
                    dict(
                        time=times[0],
                        seat=None,
                        seats=slot["seats"] or [None],
                        action=None,
                        stone=stone,
                    )
                )
                known.add(stone)
        for event in list(rnd["events"]):
            matches = []
            for index, slot in enumerate(slots):
                if (
                    index in used
                    or slot["count"] < 2
                    or slot.get("ambiguous")
                    or event["time"] <= slot["last"]
                ):
                    continue
                for obs in rows:
                    if abs(obs.time - event["time"]) > 0.01:
                        continue
                    tiles = {s.stone: s for s in obs.board}
                    new = tiles.get(event["stone"])
                    if new is None:
                        continue
                    compatible = []
                    for old in slot["anchors"]:
                        anchor = tiles.get(old.stone)
                        if anchor is None:
                            continue
                        scale = max(anchor.box[2:]) / max(old.box[2:])
                        predicted = tuple(
                            anchor.center[i] + (slot["tile"].center[i] - old.center[i]) * scale
                            for i in range(2)
                        )
                        compatible.append(math.dist(predicted, new.center) < min(new.box[2:]) * 0.6)
                    if compatible and all(compatible):
                        matches.append((index, slot, new, obs.time))
            if len(matches) == 1:
                index, slot, tile, time = matches[0]
                used.add(index)
                event["time"] = slot["time"]
                event["interval"] = slot["interval"]
                event["seats"] = slot["seats"] or event.get("seats", [event["seat"]])
                event["action"] = None
                self._entry(
                    rnd,
                    event,
                    "late_reading",
                    slot["interval"],
                    time,
                    tile.box,
                    dict(anchors=[s.stone for s in slot["anchors"]]),
                )
        for index, slot in enumerate(slots):
            if index not in used and slot["count"] >= 2:
                rnd["unresolved"].append(
                    dict(
                        time=slot["time"],
                        interval=slot["interval"],
                        stone=None,
                        candidates=[],
                        seat=None,
                        seats=slot["seats"],
                        action=None,
                        region=list(slot["tile"].box),
                        reason="ambiguous_occlusion" if slot.get("ambiguous") else "occluded",
                        anchors=[
                            dict(stone=s.stone, center=list(s.center), size=max(s.box[2:]))
                            for s in slot["anchors"]
                        ],
                    )
                )

    def _short(self, rnd, rows):
        used = {e["stone"] for e in rnd["events"]}
        used.update(s for hand in (rnd.get("remaining") or []) for s in hand)
        seen = {}
        for obs in rows:
            for tile in obs.board:
                if tile.stone not in used:
                    seen.setdefault(tile.stone, []).append((obs, tile))
        for stone, hits in seen.items():
            # Ignore isolated miscounts on the footprint of an established tile.
            first, tile = hits[0]
            earlier = [o for o in rows if first.time - 0.6 <= o.time < first.time]
            if len(hits) == 1 and any(
                s.stone in used and math.dist(s.center, tile.center) < min(tile.box[2:]) * 0.7
                for o in earlier
                for s in o.board
            ):
                continue
            seats = sorted(
                {
                    o.active
                    for o in rows
                    if first.time - 3.5 <= o.time <= first.time and o.active is not None
                }
            )
            rnd["unresolved"].append(
                dict(
                    time=first.time,
                    interval=[earlier[-1].time if earlier else first.time, first.time],
                    stone=None,
                    candidates=[stone],
                    seat=seats[-1] if seats else None,
                    seats=seats,
                    action=None,
                    region=list(tile.box),
                    reason="short_observation",
                )
            )
            rnd["unresolved"][-1]["positions"] = {s.stone: list(s.center) for s in first.board}
            rnd["unresolved"][-1]["tile_size"] = max(tile.box[2:])

    def windows(self, rounds):
        windows = []
        for rnd in rounds:
            if not rnd.get("complete"):
                continue
            if rnd.get("remaining") is not None:
                used = {e["stone"] for e in rnd["events"] if e["stone"] is not None}
                used.update(s for hand in rnd["remaining"] for s in hand)
                if len(used) < 28:
                    # A final placement may fall entirely between ordinary samples.
                    windows.append((max(rnd["start"], rnd["end"] - 1), rnd["end"]))
            for event in rnd.get("unresolved", []):
                start = max(rnd["start"], event["interval"][0] - 1)
                stop = min(rnd["end"], event["interval"][1] + 0.5)
                windows.append((start, stop))
        merged = []
        for start, stop in sorted(windows):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        return merged

    def _terminal_slots(self, rnd, rows, accounted_for):
        hits = {}
        played = {e["stone"] for e in rnd["events"]}
        for obs in rows:
            if obs.time < rnd["end"] - 1:
                continue
            for tile in obs.board:
                if tile.stone not in accounted_for:
                    hits.setdefault(tile.stone, []).append((obs, tile))
        for stone, track in hits.items():
            # A disconnected sighting is not part of the final flight.
            split = max(
                (
                    i + 1
                    for i, (a, b) in enumerate(zip(track, track[1:]))
                    if b[0].time - a[0].time > 0.15
                ),
                default=0,
            )
            track = track[split:]
            settled = [
                b
                for a, b in zip(track, track[1:])
                if b[0].time >= rnd["end"] - 0.5
                and 0 < b[0].time - a[0].time <= 0.15
                and math.dist(a[1].center, b[1].center) < 4
                and any(
                    s.stone in played and math.dist(s.center, b[1].center) < max(b[1].box[2:]) * 4
                    for s in b[0].board
                )
            ]
            if not settled:
                continue
            obs, tile = settled[-1]
            # Unknown values do not make unrelated old occlusions a match.
            # Use the installed position, not the first position in flight.
            if any(
                (not old["candidates"] or stone in old["candidates"])
                and old["interval"][0] - 1 <= obs.time <= old["interval"][1] + 0.5
                and self._fits_slot(old, tile, obs)
                for old in rnd.get("unresolved", [])
            ):
                continue
            first = track[0][0]
            seats = sorted(
                {
                    o.active
                    for o in rows
                    if first.time - 3.5 <= o.time <= first.time and o.active is not None
                }
            )
            if not seats:
                seats = [rnd.get("last_active")]
            if rnd.get("indicator_unreliable"):
                seats = [0, 1, 2, 3]
            rnd.setdefault("unresolved", []).append(
                dict(
                    time=first.time,
                    interval=[first.time, obs.time],
                    stone=None,
                    candidates=[stone],
                    seat=seats[-1],
                    seats=seats,
                    action=None,
                    region=list(tile.box),
                    reason="short_observation",
                    positions={s.stone: list(s.center) for s in obs.board},
                    tile_size=max(tile.box[2:]),
                    reading_times=[o.time for o, _ in track],
                )
            )

    def integrate(self, rounds, observations):
        for rnd in rounds:
            accounted_for = {e["stone"] for e in rnd["events"]}
            accounted_for.update(s for hand in (rnd.get("remaining") or []) for s in hand)
            rows = [
                o
                for o in observations
                if rnd["start"] <= o.time < rnd["end"] and not o.reveal and o.supported
            ]
            if rnd.get("complete") and len(accounted_for) < 28:
                self._terminal_slots(rnd, rows, accounted_for)
            for event in list(rnd.get("unresolved", [])):
                if event["reason"] == "ambiguous_occlusion":
                    continue
                later = {
                    e["stone"]: e
                    for e in rnd["events"]
                    if event["reason"] == "occluded"
                    and e["time"] > event["interval"][1] + 0.5
                    and not e.get("recovery")
                }
                hits = {}
                for obs in rows:
                    if not event["interval"][0] - 1 <= obs.time <= event["interval"][1] + 0.5:
                        continue
                    for tile in obs.board:
                        if tile.stone in accounted_for and tile.stone not in later:
                            continue
                        if event["candidates"] and tile.stone not in event["candidates"]:
                            continue
                        if obs.board and any(
                            s.stone in {e["stone"] for e in rnd["events"]}
                            and s.stone != tile.stone
                            and math.dist(s.center, tile.center) < max(tile.box[2:]) * 4
                            for s in obs.board
                        ):
                            hits.setdefault(tile.stone, []).append((obs.time, tile))
                valid = []
                for stone, track in hits.items():
                    # Flight may briefly hide the dots. Require a continuous
                    # final track; disconnected earlier sightings add no proof.
                    split = max(
                        (
                            i + 1
                            for i, (a, b) in enumerate(zip(track, track[1:]))
                            if b[0] - a[0] > 0.15
                        ),
                        default=0,
                    )
                    track = track[split:]
                    if len(track) < 3 or len({t for t, _ in track}) < 3:
                        continue
                    gaps = [b[0] - a[0] for a, b in zip(track, track[1:])]
                    if max(gaps, default=0) > 0.15:
                        continue
                    settled = any(
                        math.dist(a[1].center, b[1].center) < 4 for a, b in zip(track, track[1:])
                    )
                    terminal = rnd["end"] - track[-1][0] < 0.5
                    linked = event["reason"] == "occluded" and math.dist(
                        track[-1][1].center,
                        (
                            event["region"][0] + event["region"][2] / 2,
                            event["region"][1] + event["region"][3] / 2,
                        ),
                    ) < max(track[-1][1].box[2:])
                    retained = any(
                        o.time >= track[-1][0] and any(s.stone == stone for s in o.board)
                        for o in rows[-1:]
                    )
                    if terminal or linked or (settled and retained):
                        valid.append((stone, track))
                if len(valid) != 1:
                    continue
                stone, track = valid[0]
                last_row = next(o for o in rows if o.time == track[-1][0])
                if not self._fits_slot(event, track[-1][1], last_row):
                    continue
                alternatives = [
                    other
                    for other in rnd["unresolved"]
                    if other is not event
                    and (not other["candidates"] or stone in other["candidates"])
                    and sum(
                        other["interval"][0] - 1 <= t <= other["interval"][1] + 0.5
                        for t, _ in track
                    )
                    >= 3
                    and self._fits_slot(other, track[-1][1], last_row)
                ]
                if alternatives:
                    event["ambiguity"] = "multiple_slots"
                    continue
                # A briefly selected/cancelled stone without placement cannot
                # turn into a move merely because it was readable in flight.
                if not (
                    rnd["end"] - track[-1][0] < 0.5
                    or any(
                        math.dist(a[1].center, b[1].center) < 4 for a, b in zip(track, track[1:])
                    )
                    or event["reason"] == "occluded"
                ):
                    continue
                resolved = deepcopy(event)
                resolved["stone"] = stone
                resolved["time"] = max(event["interval"][0], track[0][0])
                if not resolved["seats"]:
                    resolved["seats"] = [rnd.get("last_active")]
                    resolved["seat"] = rnd.get("last_active")
                if rnd.get("indicator_unreliable"):
                    resolved["seats"] = [0, 1, 2, 3]
                proof_row = next(o for o in rows if o.time == track[-1][0])
                resolved["positions"] = {s.stone: list(s.center) for s in proof_row.board}
                resolved["tile_size"] = max(track[-1][1].box[2:])
                if stone in later:
                    # A native flight into the old hidden slot dates the move
                    # before its first unobstructed reading after a rearrangement.
                    for field in ("positions", "tile_size"):
                        if field in later[stone]:
                            resolved[field] = later[stone][field]
                    later[stone].update(resolved)
                    resolved = later[stone]
                else:
                    rnd["events"].append(resolved)
                accounted_for.add(stone)
                self._entry(
                    rnd,
                    resolved,
                    "animation",
                    event["interval"],
                    track[0][0],
                    track[0][1].box,
                    dict(times=[t for t, _ in track]),
                )
                rnd["unresolved"].remove(event)
            rnd["events"].sort(key=lambda e: e["time"])
            for index, event in enumerate(rnd["events"], 1):
                # Dense frames can reveal an endpoint hidden in every coarse
                # layout. Keep the most complete settled layout for side checks.
                sightings = [
                    (o, s)
                    for o in rows
                    if o.time >= event["time"]
                    for s in o.board
                    if s.stone == event["stone"]
                ]
                for (before, first), (obs, tile) in zip(sightings, sightings[1:]):
                    if (
                        0 < obs.time - before.time <= 0.15
                        and math.dist(first.center, tile.center) < 4
                        and len(obs.board) > len(event.get("positions", {}))
                    ):
                        event["positions"] = {s.stone: list(s.center) for s in obs.board}
                        event["tile_size"] = max(tile.box[2:])
                event["id"] = index
                if event.get("recovery"):
                    event["recovery"]["event_id"] = index

    def _fits_slot(self, event, tile, obs):
        region = event.get("region")
        if not region:
            return False
        center = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        visible = {s.stone: s for s in obs.board}
        predictions = []
        for anchor in event.get("anchors", []):
            current = visible.get(anchor["stone"])
            if current is not None:
                scale = max(current.box[2:]) / anchor["size"]
                predictions.append(
                    tuple(
                        current.center[i] + (center[i] - anchor["center"][i]) * scale
                        for i in range(2)
                    )
                )
        return all(
            math.dist(predicted, tile.center) < max(tile.box[2:]) * 1.2
            for predicted in (predictions or [center])
        )

    def exclusion(self, rnd, events, missing):
        if len(missing) != 1 or not all(rnd.get("remaining_confirmed", [False] * 4)):
            return False
        slots = [e for e in events if e["stone"] is None]
        if not slots:
            slots = [
                e
                for e in rnd.get("unresolved", [])
                if not e["candidates"] or missing[0] in e["candidates"]
            ]
        if not slots and rnd.get("last_active") is not None:
            # A completed reveal, confirmed empty winner and complete 27-tile
            # accounting provide evidence for the final missing placement.
            seat = rnd["last_active"]
            if not rnd["remaining"][seat]:
                slots = [
                    dict(
                        time=events[-1]["time"],
                        interval=[events[-1]["time"], rnd["end"]],
                        seat=seat,
                        seats=[seat],
                        action=None,
                        stone=None,
                    )
                ]
        if len(slots) != 1:
            return False
        played = [e["stone"] for e in events if e["stone"] is not None]
        event = slots[0]
        event["stone"] = missing[0]
        if event not in events:
            events.append(deepcopy(event))
            events.sort(key=lambda e: e["time"])
        entry = self._entry(
            rnd,
            event,
            "exclusion",
            event.get("interval", [event["time"], event["time"]]),
            evidence=dict(
                missing=missing,
                played=played,
                remaining=deepcopy(rnd["remaining"]),
            ),
        )
        if event.get("reading_times"):
            entry["evidence"]["times"] = list(event["reading_times"])
        return True
