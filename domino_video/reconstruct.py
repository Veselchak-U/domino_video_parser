"""Восстановление событий и рук без подстановки содержимого эталонного видео."""

import math
from collections import Counter
from copy import deepcopy

from .validator import GameValidator, InvalidGame


class ReconstructionError(ValueError):
    pass


class NameRecognitionError(ReconstructionError):
    pass


class ScoreRecognitionError(ReconstructionError):
    def __init__(self, round_number):
        super().__init__(f"Кон {round_number}: не прочитан итоговый счёт")
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
        seen_at = {}
        hand_history = []
        saw_empty = False
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
                seen_at = {}
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
                if candidate["count"] < 3:
                    continue
                when = first_seen[stone]
                seats = [p for t, p in history if t <= when - 0.3]
                seat = seats[-1] if seats else (obs.active if obs.active is not None else None)
                action = self._side(current["events"], tile, by_stone)
                choices = sorted({p for t, p in history if when - 3.5 <= t <= when - 0.3}) or [seat]
                current["events"].append(
                    dict(
                        time=when,
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
        return rounds

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
        for seat in range(4):
            options = Counter(tuple(sorted(row.hands[seat])) for row in rows if row.hands[seat])
            # Require a repeated stable reveal; prefer the full hand over animation fragments.
            valid = [hand for hand, count in options.items() if count >= 2]
            remaining.append(
                list(max(valid, key=lambda hand: (len(hand), options[hand]))) if valid else []
            )
        current["remaining"] = remaining
        current["last_active"] = next(
            (seat for t, seat in reversed(history) if t <= current["end"]), None
        )

    def build(self, rounds, names, variant, limit):
        if names is None or len(set(names)) != 4 or not all(names):
            raise NameRecognitionError("Не удалось прочитать четыре различных имени")
        teams = [
            dict(id="A", name="Team A", players=[dict(name=names[i], seat=i + 1) for i in [0, 2]]),
            dict(id="B", name="Team B", players=[dict(name=names[i], seat=i + 1) for i in [1, 3]]),
        ]
        candidates = [[]]
        for number, raw in enumerate(rounds, 1):
            if not raw["complete"] or raw["remaining"] is None:
                raise ReconstructionError(f"Кон {number}: неполная запись")
            if not raw.get("start_observed", True):
                raise ReconstructionError(f"Кон {number}: не записано начало розыгрыша")
            events = deepcopy(raw["events"])
            used = [event["stone"] for event in events] + [
                s for hand in raw["remaining"] for s in hand
            ]
            missing = {f"{a}-{b}" for a in range(7) for b in range(a, 7)} - set(used)
            if len(used) != len(set(used)):
                raise ReconstructionError(f"Кон {number}: повтор камня в наблюдениях")
            if len(missing) == 1:
                # Some clients remove the board before a last move has a stable frame.
                # Only one missing stone and a known last actor can be recovered.
                events.append(
                    dict(time=raw["end"], seat=raw["last_active"], action=None, stone=missing.pop())
                )
            elif missing:
                raise ReconstructionError(f"Кон {number}: не распознаны камни {sorted(missing)}")
            options = self._round_options(raw, events, names, number)
            if len(candidates) * len(options) > 128:
                raise ReconstructionError(f"Кон {number}: слишком много сочетаний реконструкций")
            candidates = [prior + [rnd] for prior in candidates for rnd in options]
            if not candidates or len(candidates) > 128:
                raise ReconstructionError(f"Кон {number}: не удаётся однозначно восстановить края")
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
                errors[0] if not valid and errors else "Несколько допустимых реконструкций"
            )
        return valid[0]

    def _round_options(self, raw, events, names, number):
        remaining = raw["remaining"]
        initial = raw.get("initial_hand")
        # Each skipped player must lack both endpoints in every still-unplayed stone.
        states = [([], None, None, [[] for _ in range(4)], [set() for _ in range(4)])]
        for index, event in enumerate(events):
            expanded = []
            oriented = (event.get("orientation") or event["stone"]) if not index else event["stone"]
            a, b = map(int, oriented.split("-"))
            for moves, ends, turn, played, forbidden in states:
                for seat in event.get("seats", [event["seat"]]):
                    if seat is None or len(played[seat]) + len(remaining[seat]) >= 7:
                        continue
                    if initial is not None and ((event["stone"] in initial) != (seat == 0)):
                        continue
                    if {a, b} & forbidden[seat]:
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
                        new_ends = list(ends) if ends else [a, b]
                        move_stone = oriented
                        if action != "start":
                            side = 0 if action == "left" else 1
                            if new_ends[side] not in (a, b):
                                continue
                            joined = new_ends[side]
                            outer = b if joined == a else a
                            move_stone = (
                                f"{outer}-{joined}" if action == "left" else f"{joined}-{outer}"
                            )
                            new_ends[side] = outer
                        new_played = deepcopy(played)
                        new_played[seat].append(event["stone"])
                        expanded.append(
                            (
                                next_moves
                                + [dict(player=names[seat], action=action, stone=move_stone)],
                                new_ends,
                                (seat + 1) % 4,
                                new_played,
                                next_forbidden,
                            )
                        )
            states = expanded
            if len(states) > 4096:
                raise ReconstructionError(f"Кон {number}: превышен предел неоднозначных событий")
        options = []
        for moves, _, _, played, _ in states:
            hands = {names[i]: sorted(remaining[i] + played[i]) for i in range(4)}
            if all(len(hand) == 7 for hand in hands.values()):
                options.append(dict(number=number, deal=hands, moves=moves))
        return options
