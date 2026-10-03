"""Восстановление событий и рук без подстановки содержимого эталонного видео."""

import math
from collections import Counter
from copy import deepcopy

from .indicator_evidence import mark_weak_indicator
from .reconstruction_counters import ahead_selection_counts, count_readings
from .reconstruction_diagnostics import event_trace, failure_message, reject
from .stone_recovery import StoneRecovery
from .validator import GameValidator, InvalidGame


class ReconstructionError(ValueError):
    def __init__(
        self,
        message,
        *,
        code="reconstruction_failed",
        round_number=None,
        event=None,
        time=None,
        details=None,
    ):
        super().__init__(message)
        self.diagnostic = dict(
            code=code,
            message=message,
            round=round_number,
            event=event,
            time=time,
            details=deepcopy(details or {}),
        )


class NameRecognitionError(ReconstructionError):
    pass


class ScoreRecognitionError(ReconstructionError):
    def __init__(self, round_number):
        super().__init__(
            f"Кон {round_number}: не прочитан итоговый счёт",
            code="score_missing",
            round_number=round_number,
        )
        self.round_number = round_number


class GameReconstructor:
    def extract(self, observations):
        rounds = []
        current = None
        pending = {}
        history = []
        reveal_rows = []
        last_board = {}
        last_reveal = False
        first_seen = {}
        placed_at = {}
        seen_at = {}
        hand_history = []
        saw_empty = False
        indicator_rows = []
        for obs in observations:
            if not obs.supported:
                continue
            if len(obs.hands[0]) == 7 and not obs.reveal:
                hand_history.append((obs.time, tuple(sorted(obs.hands[0]))))
            if obs.active is not None:
                history.append((obs.time, obs.active))
            if obs.reveal:
                if current is not None:
                    current["complete"] = True
                    if current["end"] is None:
                        current["end"] = obs.time
                    reveal_rows.append(obs)
                last_reveal = True
                indicator_rows = []
                continue
            if last_reveal:
                if current is not None:
                    self._finish(current, reveal_rows, history)
                    rounds.append(current)
                current = None
                pending = {}
                last_board = {}
                reveal_rows = []
                last_reveal = False
                first_seen = {}
                placed_at = {}
                seen_at = {}
            indicator_rows.append(obs)
            if not obs.board:
                saw_empty = True
                continue
            if current is None:
                if len(obs.board) > 2:
                    continue
                current = dict(
                    start=obs.time, end=None, complete=False, events=[], remaining=None, issues=[]
                )
                current["start_observed"] = saw_empty
                saw_empty = False
                hands = Counter(h for t, h in hand_history if obs.time - 12 < t <= obs.time)
                current["initial_hand"] = list(hands.most_common(1)[0][0]) if hands else None
                mark_weak_indicator(
                    current, [o for o in indicator_rows if o.time >= obs.time - 3.5]
                )
            mark_weak_indicator(current, [obs])
            if obs.counts is not None:
                current.setdefault("counters", []).append({"time": obs.time, "counts": obs.counts})
            by_stone = {s.stone: s for s in obs.board}
            known = {e["stone"] for e in current["events"]}
            for stone, tile in by_stone.items():
                if stone in known:
                    continue
                if obs.time - seen_at.get(stone, -100) > 2.5:
                    first_seen[stone] = obs.time
                seen_at[stone] = obs.time
                occluded = False
                for old, old_tile in last_board.items():
                    if (
                        old in known
                        and math.dist(old_tile.center, tile.center) < min(tile.box[2:]) * 0.7
                    ):
                        occluded = True
                if occluded:
                    continue
                previous = pending.get(stone)
                if previous and math.dist(previous["tile"].center, tile.center) < 4:
                    previous["count"] += 1
                    previous["tile"] = tile
                else:
                    pending[stone] = dict(time=obs.time, count=1, tile=tile)
                candidate = pending[stone]
                if candidate["count"] >= 2 and any(
                    s.stone in known and math.dist(s.center, tile.center) < max(tile.box[2:]) * 1.75
                    for s in obs.board
                ):
                    # Preserve a briefly installed tile across a later occlusion.
                    # This is a time anchor, not enough evidence to emit a move.
                    placed_at.setdefault(stone, candidate["time"])
                if candidate["count"] < 3:
                    continue
                when = min(placed_at.get(stone, first_seen[stone]), first_seen[stone])
                seats = [p for t, p in history if t <= when - 0.3]
                seat = seats[-1] if seats else (obs.active if obs.active is not None else None)
                action = self._side(current["events"], tile, by_stone)
                choices = sorted({p for t, p in history if when - 3.5 <= t <= when - 0.3}) or [seat]
                eligible = [
                    o for o in indicator_rows if o.active is not None and o.time <= when - 0.3
                ]
                mark_weak_indicator(
                    current,
                    [o for o in eligible if o.time >= when - 3.5] + eligible[-1:],
                )
                current["events"].append(
                    dict(
                        time=when,
                        confirmed_at=obs.time,
                        seat=seat,
                        seats=choices,
                        action=action,
                        stone=stone,
                        orientation="-".join(map(str, tile.values)) if action == "start" else None,
                    )
                )
                known.add(stone)
                current["events"].sort(key=lambda e: e["time"])
            for stone in list(pending):
                if stone not in by_stone:
                    del pending[stone]
            for key, tile in by_stone.items():
                if (
                    key not in last_board
                    or math.dist(last_board[key].center, tile.center) < max(tile.box[2:]) * 4
                ):
                    last_board[key] = tile
        if current:
            if reveal_rows:
                self._finish(current, reveal_rows, history)
            rounds.append(current)
        for rnd in rounds:
            self._chronological_sides(rnd, observations)
        return StoneRecovery().augment(rounds, observations)

    def _chronological_sides(self, rnd, observations):
        # Confirmation order can differ from placement order after an occlusion.
        # Derive sides again from chronological events and their actual layouts.
        prior = []
        rows = {o.time: o for o in observations}
        for event in rnd["events"]:
            event["action"] = None
            obs = rows.get(event["confirmed_at"])
            if obs is not None:
                visible = {tile.stone: tile for tile in obs.board}
                tile = visible.get(event["stone"])
                if tile is not None:
                    side = self._side(prior, tile, visible)
                    event["action"] = side
                    if side == "start":
                        event["orientation"] = "-".join(map(str, tile.values))
            prior.append(event)

    def recovery_windows(self, rounds):
        return StoneRecovery().windows(rounds)

    def integrate_recovery(self, rounds, observations):
        StoneRecovery().integrate(rounds, observations)

    def _side(self, events, tile, visible):
        if not events:
            return "start"
        chain = []
        for event in events:
            if event["action"] in ("start", "right"):
                chain.append(event["stone"])
            elif event["action"] == "left":
                chain.insert(0, event["stone"])
            else:
                return None
        if chain[0] in visible and chain[-1] in visible:
            left, right = visible[chain[0]], visible[chain[-1]]
            if len(chain) == 1:
                return "left" if tile.center[0] < left.center[0] else "right"
            return (
                "left"
                if math.dist(tile.center, left.center) < math.dist(tile.center, right.center)
                else "right"
            )
        return None

    def _finish(self, current, rows, history):
        remaining = []
        confirmed = []
        last_active = next((seat for t, seat in reversed(history) if t <= current["end"]), None)
        for seat in range(4):
            options = Counter(tuple(sorted(row.hands[seat])) for row in rows if row.hands[seat])
            # Require a repeated stable reveal; prefer the full hand over animation fragments.
            valid = [hand for hand, count in options.items() if count >= 2]
            remaining.append(
                list(max(valid, key=lambda hand: (len(hand), options[hand]))) if valid else []
            )
            valid_area = sum(
                row.reveal_valid is not None and row.reveal_valid[seat] for row in rows
            )
            empty = sum(not row.hands[seat] for row in rows) >= 2
            confirmed.append(
                bool(valid)
                or (
                    empty
                    and last_active == seat
                    and (valid_area >= 2 or all(row.reveal_valid is None for row in rows))
                )
            )
        current["remaining"] = remaining
        current["remaining_confirmed"] = confirmed
        current["last_active"] = next(
            (seat for t, seat in reversed(history) if t <= current["end"]), None
        )

    def build(self, rounds, names, variant, limit):
        if names is None or len(set(names)) != 4 or not all(names):
            raise NameRecognitionError(
                "Не удалось прочитать четыре различных имени", code="invalid_player_names"
            )
        teams = [
            dict(id="A", name="Team A", players=[dict(name=names[i], seat=i + 1) for i in [0, 2]]),
            dict(id="B", name="Team B", players=[dict(name=names[i], seat=i + 1) for i in [1, 3]]),
        ]
        candidates = [[]]
        for number, raw in enumerate(rounds, 1):
            if not raw["complete"] or raw["remaining"] is None:
                raise ReconstructionError(
                    f"Кон {number}: неполная запись",
                    code="incomplete_round",
                    round_number=number,
                    details=dict(
                        complete=raw["complete"], start=raw.get("start"), end=raw.get("end")
                    ),
                )
            if not raw.get("start_observed", True):
                raise ReconstructionError(
                    f"Кон {number}: не записано начало розыгрыша",
                    code="missing_start",
                    round_number=number,
                    details=dict(first_observed=raw.get("start")),
                )
            events = deepcopy(raw["events"])
            used = [event["stone"] for event in events] + [
                s for hand in raw["remaining"] for s in hand
            ]
            used = [stone for stone in used if stone is not None]
            missing = {f"{a}-{b}" for a in range(7) for b in range(a, 7)} - set(used)
            if len(used) != len(set(used)):
                raise ReconstructionError(
                    f"Кон {number}: повтор камня в наблюдениях",
                    code="duplicate_stones",
                    round_number=number,
                    details=dict(repeated={s: n for s, n in Counter(used).items() if n > 1}),
                )
            if missing:
                if not all(raw.get("remaining_confirmed", [False] * 4)):
                    last = raw.get("last_active")
                    raise ReconstructionError(
                        f"Кон {number}: не подтверждены конечные остатки",
                        code="unconfirmed_remaining",
                        round_number=number,
                        time=raw.get("end"),
                        details=dict(
                            unconfirmed_seats=[
                                i + 1
                                for i, ok in enumerate(raw.get("remaining_confirmed", [False] * 4))
                                if not ok
                            ],
                            remaining=raw["remaining"],
                            missing=sorted(missing),
                            accounted_count=len(set(used)),
                            last_active=None if last is None else last + 1,
                        ),
                    )
                if not StoneRecovery().exclusion(raw, events, sorted(missing)):
                    raise ReconstructionError(
                        f"Кон {number}: не распознаны камни {sorted(missing)}",
                        code="missing_stones",
                        round_number=number,
                        details=dict(
                            missing=sorted(missing),
                            unresolved=raw.get("unresolved", [])[:8],
                            omitted_unresolved=max(0, len(raw.get("unresolved", [])) - 8),
                        ),
                    )
            if any(e["stone"] is None for e in events) or raw.get("issues"):
                raise ReconstructionError(
                    f"Кон {number}: противоречивые или неизвестные события",
                    code="unresolved_events",
                    round_number=number,
                    details=dict(
                        issues=raw.get("issues", []),
                        unknown_events=[i + 1 for i, e in enumerate(events) if e["stone"] is None],
                    ),
                )
            for index, event in enumerate(events, 1):
                for entry in raw.get("stone_recovery", []):
                    if entry["stone"] == event["stone"]:
                        entry["event_id"] = index
            trace = []
            options = self._round_options(raw, events, names, number, trace)
            if len(candidates) * len(options) > 128:
                raise ReconstructionError(
                    f"Кон {number}: слишком много сочетаний реконструкций",
                    code="reconstruction_combination_limit",
                    round_number=number,
                    details=dict(
                        prior_candidates=len(candidates), round_options=len(options), limit=128
                    ),
                )
            candidates = [prior + [rnd] for prior in candidates for rnd in options]
            if not candidates or len(candidates) > 128:
                failure = next(
                    (e for e in trace if e["states_before"] and not e["states_after"]), None
                )
                raise ReconstructionError(
                    failure_message(number, failure),
                    code="no_consistent_sequence",
                    round_number=number,
                    event=failure["event"] if failure else None,
                    time=failure["time"] if failure else None,
                    details=dict(first_failure=failure, events=trace),
                )
        valid = []
        errors = []
        validator = GameValidator()
        for candidate in candidates:
            try:
                game = validator.calculate(teams, candidate, variant, limit)
                validator.validate(game)
                valid.append(game)
            except InvalidGame as error:
                errors.append(str(error))
        if len(valid) != 1:
            raise ReconstructionError(
                errors[0] if not valid and errors else "Несколько допустимых реконструкций",
                code="all_candidates_invalid" if not valid else "multiple_valid_reconstructions",
                details=dict(
                    candidate_count=len(candidates),
                    valid_count=len(valid),
                    validation_errors=errors[:5],
                    omitted_errors=max(0, len(errors) - 5),
                ),
            )
        for raw, rnd in zip(rounds, valid[0]["rounds"]):
            for entry in raw.get("stone_recovery", []):
                entry["status"] = "rules_validated"
                matching = [
                    m
                    for m in rnd["moves"]
                    if m.get("stone")
                    and "-".join(map(str, sorted(map(int, m["stone"].split("-")))))
                    == entry["stone"]
                ]
                if matching:
                    entry["seat"] = names.index(matching[0]["player"]) + 1
        return valid[0]

    def _round_options(self, raw, events, names, number, diagnostics=None):
        remaining = raw["remaining"]
        initial = raw.get("initial_hand")
        # Each skipped player must lack both endpoints in every still-unplayed stone.
        states = [([], None, None, [[] for _ in range(4)], [set() for _ in range(4)], Counter())]
        for index, event in enumerate(events):
            entry = event_trace(event, index, len(states))
            simultaneous = bool(index and abs(event["time"] - events[index - 1]["time"]) < 0.01)
            if diagnostics is not None:
                diagnostics.append(entry)
            expanded = []
            oriented = (event.get("orientation") or event["stone"]) if not index else event["stone"]
            a, b = map(int, oriented.split("-"))
            for moves, ends, turn, played, forbidden, selected in states:
                for seat in event.get("seats", [event["seat"]]):
                    if seat is None or len(played[seat]) + len(remaining[seat]) >= 7:
                        if seat is None:
                            reject(entry, "unknown_player")
                        else:
                            reject(
                                entry,
                                "hand_capacity_exceeded",
                                seat=seat + 1,
                                played=len(played[seat]),
                                remaining=len(remaining[seat]),
                            )
                        continue
                    if initial is not None and ((event["stone"] in initial) != (seat == 0)):
                        reject(entry, "initial_hand_conflict", seat=seat + 1, initial_hand=initial)
                        continue
                    conflicts = [
                        dict(seat=p + 1, selected_player=seat + 1, count=count)
                        for p, count in selected.items()
                        if count >= 2 and p != seat
                    ]
                    if conflicts and not (
                        initial is not None and event["stone"] in initial and seat == 0
                    ):
                        reject(
                            entry,
                            "hand_counter_conflict",
                            seat=seat + 1,
                            phase="next_selection",
                            conflicts=conflicts,
                        )
                        continue
                    if {a, b} & forbidden[seat]:
                        reject(
                            entry,
                            "prior_pass_conflict",
                            seat=seat + 1,
                            forbidden=sorted(forbidden[seat]),
                            ends=ends,
                        )
                        continue
                    next_moves = deepcopy(moves)
                    next_forbidden = deepcopy(forbidden)
                    cursor = turn
                    possible = True
                    while cursor is not None and cursor != seat:
                        hand = (
                            (set(initial) - set(played[0]))
                            if cursor == 0 and initial
                            else remaining[cursor]
                        )
                        if any(set(map(int, s.split("-"))) & set(ends) for s in hand):
                            reject(
                                entry,
                                "illegal_pass",
                                seat=cursor + 1,
                                ends=ends,
                                blocking_stones=sorted(
                                    s for s in hand if set(map(int, s.split("-"))) & set(ends)
                                ),
                            )
                            possible = False
                            break
                        next_forbidden[cursor].update(ends)
                        next_moves.append(dict(player=names[cursor], action="pass"))
                        cursor = (cursor + 1) % 4
                    if not possible:
                        continue
                    actions = (
                        ["start"]
                        if not index
                        else ([event["action"]] if event["action"] else ["left", "right"])
                    )
                    for action in actions:
                        positions = event.get("positions", {})
                        chain = []
                        for move in moves:
                            if move["action"] == "start" or move["action"] == "right":
                                chain.append(
                                    "-".join(map(str, sorted(map(int, move["stone"].split("-")))))
                                )
                            elif move["action"] == "left":
                                chain.insert(
                                    0,
                                    "-".join(map(str, sorted(map(int, move["stone"].split("-"))))),
                                )
                        linked_sides = (
                            {
                                side
                                for side, endpoint in (("left", chain[0]), ("right", chain[-1]))
                                for link in event.get("placement_links", [])
                                if link["stone"] == endpoint
                            }
                            if len(chain) > 1
                            else set()
                        )
                        if (
                            not event["action"]
                            and not simultaneous
                            and len(linked_sides) == 1
                            and action not in linked_sides
                        ):
                            reject(
                                entry,
                                "geometry_conflict",
                                seat=seat + 1,
                                action=action,
                                observed_side=next(iter(linked_sides)),
                                evidence="placement_contact",
                            )
                            continue
                        if (
                            not event["action"]
                            and not simultaneous
                            and action != "start"
                            and len(chain) > 1
                            and all(s in positions for s in [chain[0], chain[-1], event["stone"]])
                        ):
                            left_distance = math.dist(
                                positions[event["stone"]], positions[chain[0]]
                            )
                            right_distance = math.dist(
                                positions[event["stone"]], positions[chain[-1]]
                            )
                            if (
                                min(left_distance, right_distance)
                                < event.get("tile_size", 100) * 1.75
                                and abs(left_distance - right_distance) > 20
                            ):
                                observed_side = (
                                    "left" if left_distance < right_distance else "right"
                                )
                                if action != observed_side:
                                    reject(
                                        entry,
                                        "geometry_conflict",
                                        seat=seat + 1,
                                        action=action,
                                        observed_side=observed_side,
                                        left_distance=left_distance,
                                        right_distance=right_distance,
                                        tile_size=event.get("tile_size", 100),
                                    )
                                    continue
                        new_ends = list(ends) if ends else [a, b]
                        move_stone = oriented
                        if action != "start":
                            side = 0 if action == "left" else 1
                            if new_ends[side] not in (a, b):
                                reject(
                                    entry,
                                    "endpoint_mismatch",
                                    seat=seat + 1,
                                    action=action,
                                    ends=new_ends,
                                    oriented_stone=oriented,
                                )
                                continue
                            joined = new_ends[side]
                            outer = b if joined == a else a
                            move_stone = (
                                f"{outer}-{joined}" if action == "left" else f"{joined}-{outer}"
                            )
                            new_ends[side] = outer
                        new_played = deepcopy(played)
                        new_played[seat].append(event["stone"])
                        next_selected = Counter()
                        if raw.get("indicator_unreliable") or len(set(event.get("seats", []))) > 1:
                            stop = (
                                events[index + 1]["time"] if index + 1 < len(events) else raw["end"]
                            )
                            counters = [
                                o
                                for o in raw.get("counters", [])
                                if event["time"] + 0.3 < o["time"] < stop - 0.3
                            ]
                            readings = count_readings(
                                counters,
                                new_played,
                                events[index + 1] if index + 1 < len(events) else None,
                            )
                            next_selected = ahead_selection_counts(
                                counters,
                                new_played,
                                events[index + 1] if index + 1 < len(events) else None,
                            )
                            if (
                                any(
                                    count >= 2 and value != 7 - len(new_played[p])
                                    for p, reading in enumerate(readings)
                                    if p != 0
                                    for value, count in reading.items()
                                )
                                and not simultaneous
                            ):
                                conflicts = [
                                    dict(
                                        seat=p + 1,
                                        expected=7 - len(new_played[p]),
                                        observed=value,
                                        count=count,
                                    )
                                    for p, reading in enumerate(readings)
                                    if p != 0
                                    for value, count in reading.items()
                                    if count >= 2 and value != 7 - len(new_played[p])
                                ]
                                reject(
                                    entry,
                                    "hand_counter_conflict",
                                    seat=seat + 1,
                                    conflicts=conflicts,
                                    interval=[event["time"] + 0.3, stop - 0.3],
                                )
                                continue
                        expanded.append(
                            (
                                next_moves
                                + [dict(player=names[seat], action=action, stone=move_stone)],
                                new_ends,
                                (seat + 1) % 4,
                                new_played,
                                next_forbidden,
                                next_selected,
                            )
                        )
            states = expanded
            entry["states_after"] = len(states)
            if len(states) > 4096:
                raise ReconstructionError(
                    f"Кон {number}: превышен предел неоднозначных событий",
                    code="search_state_limit",
                    round_number=number,
                    event=index + 1,
                    time=event["time"],
                    details=dict(count=len(states), limit=4096, events=diagnostics or [entry]),
                )
        options = []
        final = dict(
            event=None,
            time=None,
            stone=None,
            states_before=len(states),
            states_after=0,
            rejections={},
        )
        for moves, _, _, played, _, _ in states:
            hands = {names[i]: sorted(remaining[i] + played[i]) for i in range(4)}
            if all(len(hand) == 7 for hand in hands.values()):
                options.append(dict(number=number, deal=hands, moves=moves))
            else:
                reject(final, "incomplete_deal", hand_sizes=[len(hands[name]) for name in names])
        final["states_after"] = len(options)
        if diagnostics is not None and final["rejections"]:
            diagnostics.append(final)
        return options
