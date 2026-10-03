"""Evidence for missed moves; no video I/O or publication of files."""

import math
from collections import Counter
from copy import deepcopy
from itertools import groupby

from .indicator_evidence import mark_weak_indicator, preserve_player_candidates


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
            self._delayed_layouts(rnd, rows)
            self._turn_windows(rnd, rows)
            preserve_player_candidates(rnd)
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
        seen_times = set()
        for obs in rows:
            if obs.time in seen_times:
                continue
            seen_times.add(obs.time)
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
                # A dragged tile can disappear from the hand for several frames
                # and then return. Retract only our hand-only hypothesis; an
                # independently confirmed board placement remains a conflict.
                if set(before) < set(hand) and len(hand) == len(before) + 1:
                    cancelled = []
                    for event in rnd["events"]:
                        proof = event.get("recovery", {})
                        evidence = proof.get("evidence", {})
                        if (
                            event["time"] < first
                            and event.get("confirmed_at") is None
                            and proof.get("method") == "hand_difference"
                            and evidence.get("before") == list(hand)
                            and evidence.get("after") == list(before)
                        ):
                            cancelled.append(event)
                    for event in cancelled:
                        rnd["events"].remove(event)
                        rnd["stone_recovery"].remove(event["recovery"])
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
        preserve_player_candidates(rnd)

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
            windows.extend((start, stop) for start, stop, _ in self._early_groups(rnd))
            windows.extend(tuple(window) for window in rnd.get("transition_windows", []))
            if rnd.get("remaining") is not None:
                used = {e["stone"] for e in rnd["events"] if e["stone"] is not None}
                used.update(s for hand in rnd["remaining"] for s in hand)
                if len(used) < 28:
                    # Several final placements can precede the clearing. Include
                    # their incoming flights and the indicator before each one.
                    previous = max(
                        (e["time"] for e in rnd["events"] if e.get("time") is not None),
                        default=rnd["end"] - 0.5,
                    )
                    windows.append((max(rnd["start"], previous - 0.5), rnd["end"]))
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

    def _turn_windows(self, rnd, rows):
        if not rnd.get("complete"):
            return
        runs = []
        for obs in rows:
            if obs.active is None:
                continue
            if not runs or runs[-1][0].active != obs.active or obs.time - runs[-1][-1].time > 3:
                runs.append([])
            runs[-1].append(obs)
        windows = []
        for before, after in zip(runs, runs[1:]):
            if (
                len({o.time for o in before}) < 2
                or len({o.time for o in after}) < 2
                or after[0].time - before[-1].time > 3
            ):
                continue
            if any(
                before[0].time - 0.3 <= event["time"] <= after[0].time + 0.3
                for event in rnd["events"]
            ):
                continue
            windows.append(
                [max(rnd["start"], before[-1].time - 0.5), min(rnd["end"], after[0].time + 0.5)]
            )
        rnd["transition_windows"] = windows

    def _incoming(self, rnd, rows, track):
        """Require an arriving flight, three settled frames and an established neighbour."""
        if len({o.time for o, _ in track}) < 5 or self._old_tile_track(
            rnd, rows, [(o.time, s) for o, s in track]
        ):
            return False
        first, tile = track[0]
        _, final = track[-1]
        residuals = [math.dist(s.center, final.center) for _, s in track]
        if not (
            residuals[0] > min(final.box[2:]) * 0.5
            and math.dist(tile.center, track[2][1].center) > 4
            and all(b <= a + 4 for a, b in zip(residuals, residuals[1:]))
            and all(d < 4 for d in residuals[-3:])
        ):
            return False
        known = {e["stone"] for e in rnd["events"] if e["time"] < first.time}
        known.discard(tile.stone)
        neighbours = [
            {
                s.stone
                for s in o.board
                if s.stone in known and math.dist(s.center, t.center) < max(t.box[2:]) * 1.75
            }
            for o, t in track[-3:]
        ]
        return bool(set.intersection(*neighbours))

    def _transition_placements(self, rnd, rows):
        windows = []
        for start, stop in sorted(rnd.get("transition_windows", [])):
            if windows and start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(stop, windows[-1][1]))
            else:
                windows.append((start, stop))
        if not windows:
            return
        events = {e["stone"]: e for e in rnd["events"]}
        remaining = {s for hand in (rnd.get("remaining") or []) for s in hand}
        flights = {}
        for start, stop in windows:
            hits = {}
            for obs in rows:
                if not start <= obs.time <= stop:
                    continue
                for tile in obs.board:
                    if tile.stone in remaining or tile.stone is None:
                        continue
                    existing = events.get(tile.stone)
                    if existing and existing["time"] <= start:
                        continue
                    hits.setdefault(tile.stone, []).append((obs, tile))
            for stone, sightings in hits.items():
                tracks = []
                for reading in sightings:
                    if not tracks or not 0 < reading[0].time - tracks[-1][-1][0].time <= 0.15:
                        tracks.append([])
                    tracks[-1].append(reading)
                for track in tracks:
                    if self._incoming(rnd, rows, track):
                        flights.setdefault(stone, {})[track[0][0].time] = track
        for stone, choices in sorted(flights.items(), key=lambda item: min(item[1])):
            if len(choices) != 1:
                continue
            track = next(iter(choices.values()))
            first, tile = track[0]
            existing = events.get(stone)
            if existing and first.time >= existing["time"] - 0.5:
                continue
            # An independently proven arrival is conflicting evidence, not a
            # late stationary reading that this path may replace.
            if (
                existing
                and existing.get("recovery", {}).get("evidence", {}).get("motion") == "incoming"
            ):
                continue
            seats = sorted(
                {
                    o.active
                    for o in rows
                    if first.time - 3.5 <= o.time <= first.time - 0.3 and o.active is not None
                }
            )
            if rnd.get("indicator_unreliable"):
                seats = [0, 1, 2, 3]
            if not seats:
                continue
            event = existing if existing is not None else dict(stone=stone)
            late = event.get("time")
            event.update(
                time=first.time,
                seats=seats,
                seat=seats[0] if len(seats) == 1 else None,
                action=None,
                positions={s.stone: list(s.center) for s in track[-1][0].board},
                tile_size=max(track[-1][1].box[2:]),
            )
            if existing is None:
                rnd["events"].append(event)
                events[stone] = event
            self._entry(
                rnd,
                event,
                "animation",
                [first.time, track[-1][0].time],
                first.time,
                tile.box,
                dict(times=[o.time for o, _ in track], late_reading=late, motion="incoming"),
            )

    def _delayed_layouts(self, rnd, rows):
        events = sorted(rnd["events"], key=lambda e: e["time"])
        for previous, event in zip(events, events[1:]):
            if event.get("recovery", {}).get("method") == "animation":
                continue
            start, stop = previous["time"], event["time"]
            # A layout change well before the next reading can hide a new tile.
            # Ignore the first half-second of the preceding placement itself.
            interval = [o for o in rows if start + 0.5 < o.time < stop - 0.5]
            known = {e["stone"] for e in events if e["time"] <= start}
            for before, after in zip(interval, interval[1:]):
                a = {s.stone: s for s in before.board}
                b = {s.stone: s for s in after.board}
                shared = known & a.keys() & b.keys()
                # Three independent established tiles moving by a quarter of
                # their long side distinguish chain rearrangement from jitter.
                moved = [
                    stone
                    for stone in shared
                    if math.dist(a[stone].center, b[stone].center) > max(a[stone].box[2:]) * 0.25
                ]
                if len(moved) >= 3:
                    event["early_reading_interval"] = [start, stop]
                    break

    def _early_groups(self, rnd):
        groups = self._coincident_groups(rnd)
        confirmed = {}
        for event in rnd["events"]:
            if event.get("confirmed_at") is not None:
                confirmed.setdefault(event["confirmed_at"], []).append(event)
        for stop, events in confirmed.items():
            if len(events) < 2:
                continue
            first = min(e["time"] for e in events)
            prior = [e for e in rnd["events"] if e["time"] < first]
            previous = max(prior, key=lambda e: e["time"]) if prior else None
            start = (
                previous.get("early_reading_interval", [previous["time"]])[0]
                if previous
                else rnd["start"]
            )
            start = min(start, *(e.get("early_reading_interval", [first])[0] for e in events))
            groups.append((start, stop, events))
        grouped = {id(e) for _, _, events in groups for e in events}
        for event in rnd["events"]:
            if id(event) in grouped or event.get("recovery", {}).get("method") == "animation":
                continue
            interval = event.get("early_reading_interval")
            if interval and interval[0] < event["time"]:
                groups.append((interval[0], event["time"], [event]))
        return sorted(groups, key=lambda group: group[1])

    def _coincident_groups(self, rnd):
        groups = []
        previous = None
        timed = [e for e in rnd["events"] if e.get("time") is not None]
        for time, items in groupby(sorted(timed, key=lambda e: e["time"]), key=lambda e: e["time"]):
            events = list(items)
            if len(events) > 1 and previous is not None:
                groups.append((previous, time, events))
            previous = min(e.get("early_reading_interval", [time])[0] for e in events)
        return groups

    def _earlier_placements(self, rnd, rows):
        groups = self._early_groups(rnd)
        stationary_members = {
            id(event) for _, _, events in self._coincident_groups(rnd) for event in events
        }
        stationary_members.update(
            id(event) for event in rnd["events"] if event.get("early_reading_interval")
        )
        # A requested interval can contain several hidden placements whose
        # eventual readings have different times. Keep the original interval
        # after its trigger has been recovered and inspect other values that
        # actually recur in those already decoded frames.
        intervals = {(start, stop) for start, stop, _ in groups}
        intervals.update(
            tuple(e["early_reading_interval"])
            for e in rnd["events"]
            if e.get("early_reading_interval")
        )
        scopes = {}
        for start, stop, events in groups:
            for event in events:
                old = scopes.get(id(event))
                scopes[id(event)] = [
                    min(start, old[0]) if old else start,
                    max(stop, old[1]) if old else stop,
                    event,
                    id(event) in stationary_members,
                ]
        for start, stop in sorted(intervals):
            readings = Counter(s.stone for o in rows if start <= o.time < stop for s in o.board)
            for event in rnd["events"]:
                if event["time"] < stop or readings[event["stone"]] < 3:
                    continue
                old = scopes.get(id(event))
                scopes[id(event)] = [
                    min(start, old[0]) if old else start,
                    max(stop, old[1]) if old else stop,
                    event,
                    bool(old and old[3]),
                ]
        groups = sorted(scopes.values(), key=lambda scope: scope[2]["time"])
        for start, stop, event, original_member in groups:
            if event.get("recovery", {}).get("method") == "animation":
                continue
            late_reading = event["time"]
            tracks = []
            for obs in rows:
                if not start <= obs.time < min(stop, late_reading - 0.5):
                    continue
                tiles = [s for s in obs.board if s.stone == event["stone"]]
                if len(tiles) != 1:
                    continue
                if not tracks or not 0 < obs.time - tracks[-1][-1][0].time <= 0.15:
                    tracks.append([])
                tracks[-1].append((obs, tiles[0]))
            valid = []
            for track in tracks:
                if len(track) < 3:
                    continue
                if self._old_tile_track(rnd, rows, [(o.time, s) for o, s in track]):
                    continue
                first, tile = track[0]
                last, final = track[-1]
                positions = event.get("positions", {})
                if event["stone"] not in positions:
                    continue
                known = {e["stone"] for e in rnd["events"] if e["time"] < first.time}
                linked = False
                moving = False
                proof = track
                for neighbor in sorted(known & positions.keys()):
                    if (
                        math.dist(positions[event["stone"]], positions[neighbor])
                        >= event.get("tile_size", 100) * 1.75
                    ):
                        continue
                    fragments = []
                    fragment = []
                    for obs, reading in track:
                        anchors = [s for s in obs.board if s.stone == neighbor]
                        if len(anchors) != 1:
                            if fragment:
                                fragments.append(fragment)
                                fragment = []
                            continue
                        fragment.append(
                            (obs, reading, math.dist(reading.center, anchors[0].center))
                        )
                    if fragment:
                        fragments.append(fragment)
                    final_anchors = [s for s in last.board if s.stone == neighbor]
                    residuals = [math.dist(s.center, final.center) for _, s in track]
                    # A flight can approach tangentially to its neighbor.
                    # In that case require arrival at a stable final position,
                    # not just an arbitrary decrease of inter-tile distance.
                    arriving = (
                        len(track) >= 5
                        and len(final_anchors) == 1
                        and sum(len(f) for f in fragments) >= 3
                        and math.dist(final.center, final_anchors[0].center)
                        < max(final.box[2:]) * 1.75
                        and residuals[0] > min(final.box[2:]) * 0.5
                        and math.dist(track[0][1].center, track[2][1].center) > 4
                        and all(b <= a + 4 for a, b in zip(residuals, residuals[1:]))
                        and all(d < 4 for d in residuals[-3:])
                    )
                    if arriving:
                        linked = moving = True
                        break
                    supported = []
                    for fragment in fragments:
                        if len(fragment) < 3:
                            continue
                        last_tile = fragment[-1][1]
                        distances = [d for _, _, d in fragment]
                        if distances[-1] >= max(last_tile.box[2:]) * 1.75:
                            continue
                        settled = all(
                            math.dist(fragment[0][1].center, s.center) < 4 for _, s, _ in fragment
                        )
                        incoming = distances[0] - distances[-1] > min(
                            last_tile.box[2:]
                        ) * 0.5 and all(b <= a + 4 for a, b in zip(distances, distances[1:]))
                        if settled or incoming:
                            supported.append((fragment, incoming))
                    if len(supported) == 1:
                        linked = True
                        fragment, moving = supported[0]
                        proof = [(o, s) for o, s, _ in fragment]
                        break
                first = proof[0][0]
                seats = sorted(
                    {
                        o.active
                        for o in rows
                        if first.time - 3.5 <= o.time <= first.time - 0.3 and o.active is not None
                    }
                )
                if rnd.get("indicator_unreliable"):
                    seats = [0, 1, 2, 3]
                if linked and seats and (original_member or moving):
                    valid.append((proof, seats, moving))
            flights = [entry for entry in valid if entry[2]]
            if len(flights) == 1 and all(
                entry is flights[0] or entry[0][0][0].time > flights[0][0][-1][0].time
                for entry in valid
            ):
                # Later stationary readings of the same unique domino are
                # confirmation of the one flight, not competing placements.
                valid = flights
            if len(valid) != 1:
                continue
            track, seats, _ = valid[0]
            first, tile = track[0]
            event.update(
                time=first.time,
                seats=seats,
                seat=seats[0] if len(seats) == 1 else None,
                action=None,
            )
            self._entry(
                rnd,
                event,
                "animation",
                [first.time, track[-1][0].time],
                first.time,
                tile.box,
                dict(times=[o.time for o, _ in track], late_reading=late_reading),
            )

    def _terminal_slots(self, rnd, rows, accounted_for):
        hits = {}
        played = {e["stone"] for e in rnd["events"]}
        for obs in rows:
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
            # The final second may contain the continuation of an older short
            # reading. Keep the continuous track back to its actual beginning.
            # Unknown values alone still do not link unrelated occlusions.
            if any(
                (not old["candidates"] or stone in old["candidates"])
                and any(
                    old["interval"][0] - 1 <= reading.time <= old["interval"][1] + 0.5
                    and 0 < reading.time - before.time <= 0.15
                    and math.dist(previous.center, placed.center) < 4
                    and self._fits_slot(old, placed, reading)
                    for (before, previous), (reading, placed) in zip(track, track[1:])
                )
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

    def _old_tile_track(self, rnd, rows, track):
        """Reject a temporary value change inside one existing tile's trajectory."""
        known = {e["stone"] for e in rnd["events"] if e["time"] < track[0][0]}
        known.discard(track[0][1].stone)
        if not known:
            return False

        def adjacent(first, second):
            # Reuse native-track continuity and the narrow slot-match tolerance.
            # Several readings on both sides are required; one footprint is not proof.
            return (
                0 < abs(first[0] - second[0]) <= 0.15
                and math.dist(first[1].center, second[1].center)
                < min(*first[1].box[2:], *second[1].box[2:]) * 0.6
            )

        if not all(adjacent(a, b) for a, b in zip(track, track[1:])):
            return False
        identities = []
        for boundary, reverse in [(track[0], True), (track[-1], False)]:
            surrounding = [
                o for o in rows if (o.time < boundary[0] if reverse else o.time > boundary[0])
            ]
            surrounding.sort(key=lambda o: o.time, reverse=reverse)
            previous = boundary
            stones = []
            for obs in surrounding[:3]:
                candidates = [s for s in obs.board if adjacent(previous, (obs.time, s))]
                if len(candidates) != 1 or candidates[0].stone not in known:
                    return False
                stones.append(candidates[0].stone)
                previous = obs.time, candidates[0]
            if len(stones) != 3 or len(set(stones)) != 1:
                return False
            identities.append(stones[0])
        return identities[0] == identities[1]

    def integrate(self, rounds, observations):
        for number, rnd in enumerate(rounds):
            accounted_for = {e["stone"] for e in rnd["events"]}
            accounted_for.update(s for hand in (rnd.get("remaining") or []) for s in hand)
            rows = [
                o
                for o in observations
                if rnd["start"] <= o.time < rnd["end"] and not o.reveal and o.supported
            ]
            history_start = max(
                rnd["start"] - 3.5,
                rounds[number - 1]["end"] if number else float("-inf"),
            )
            mark_weak_indicator(
                rnd, [o for o in observations if history_start < o.time < rnd["end"]]
            )
            preserve_player_candidates(rnd)
            self._earlier_placements(rnd, rows)
            self._transition_placements(rnd, rows)
            accounted_for.update(e["stone"] for e in rnd["events"])
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
                    if self._old_tile_track(rnd, rows, track):
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
            preserve_player_candidates(rnd)
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
        empty_proof = None
        if not slots:
            # A completed reveal, confirmed empty winner and complete 27-tile
            # accounting provide evidence for the final missing placement.
            seat = rnd.get("last_active")
            proofs = [
                p
                for p in rnd.get("remaining_evidence", [])
                if p.get("method") == "zero_reveal_points"
            ]
            if proofs:
                empty_proof = self._final_empty_evidence(rnd, events, missing, proofs)
                if empty_proof is None:
                    return False
                seat = empty_proof["seat"] - 1
            if seat is not None and not rnd["remaining"][seat]:
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
        if empty_proof is not None:
            entry["evidence"]["remaining_evidence"] = [deepcopy(empty_proof)]
        return True

    @staticmethod
    def _final_empty_evidence(rnd, events, missing, proofs):
        """Confirm a final slot independently of a possibly stale turn indicator.

        The zero proof certifies emptiness only. The caller still reconstructs
        and validates the unique complete game, including immediate termination.
        """
        hands = rnd["remaining"]
        known = [event["stone"] for event in events] + [stone for hand in hands for stone in hand]
        deck = {f"{a}-{b}" for a in range(7) for b in range(a, 7)}
        if (
            not rnd.get("complete")
            or rnd.get("end") is None
            or not events
            or rnd.get("unresolved")
            or len(hands) != 4
            or len(rnd["remaining_confirmed"]) != 4
            or sum(not hand for hand in hands) != 1
            or len(known) != 27
            or len(set(known)) != 27
            or set(known) != deck - set(missing)
            or len(proofs) != 1
        ):
            return None
        proof = proofs[0]
        seat = proof.get("seat")
        times = proof.get("times", [])
        points = proof.get("points", [])
        if (
            type(seat) is not int
            or not 1 <= seat <= 4
            or hands[seat - 1]
            or len(times) < 2
            or len(set(times)) < 2
            or len(times) != len(points)
            or any(type(point) is not int or point != 0 for point in points)
            or any(not math.isfinite(time) or time < rnd["end"] for time in times)
        ):
            return None
        return proof
