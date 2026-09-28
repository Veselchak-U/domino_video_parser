"""Структурная и семантическая проверка полного журнала."""

import json
from copy import deepcopy
from importlib.resources import files

from jsonschema import Draft202012Validator

from .rules import ScoringRules


class InvalidGame(ValueError):
    pass


class GameValidator:
    def __init__(self):
        schema = json.loads(
            files("domino_video").joinpath("schema.json").read_text(encoding="utf-8")
        )
        self._schema = Draft202012Validator(schema)
        self._rules = ScoringRules()

    def validate(self, document: dict) -> dict:
        errors = sorted(self._schema.iter_errors(document), key=lambda e: str(e.path))
        if errors:
            error = errors[0]
            raise InvalidGame(f"Структура JSON {list(error.path)}: {error.message}")
        calculated = self.calculate(
            document["teams"], document["rounds"], document["fishVariant"], document["scoreLimit"]
        )
        for actual, expected in zip(document["rounds"], calculated["rounds"]):
            if actual["result"] != expected["result"]:
                raise InvalidGame(
                    f"Кон {actual['number']}: результат расходится с правилами; ожидается {expected['result']}"
                )
        if document["result"] != calculated["result"]:
            raise InvalidGame("Итог партии расходится с правилами")
        return document

    def calculate(self, teams, rounds, variant, limit):
        if (
            variant not in ("withEggs", "withoutEggs")
            or type(limit) is not int
            or limit not in (50, 101)
        ):
            raise InvalidGame("Неизвестный вариант или лимит")
        if len(teams) != 2:
            raise InvalidGame("Требуются две команды")
        names, team_ids, seats, team_for = set(), set(), {}, {}
        for team in teams:
            if team["name"] in names or team["id"] in team_ids or not team["name"].strip():
                raise InvalidGame("Команды должны иметь различные непустые имена и ID")
            names.add(team["name"])
            team_ids.add(team["id"])
            positions = []
            if len(team["players"]) != 2:
                raise InvalidGame("В команде должно быть два игрока")
            for player in team["players"]:
                name, seat = player["name"], player["seat"]
                if (
                    not name.strip()
                    or name in team_for
                    or seat in seats
                    or type(seat) is not int
                    or seat not in range(1, 5)
                ):
                    raise InvalidGame("Имена игроков и места должны быть различными")
                seats[seat], team_for[name] = name, team["name"]
                positions.append(seat)
            if abs(positions[0] - positions[1]) != 2:
                raise InvalidGame("Партнёры должны сидеть напротив")
        order = [seats[i] for i in range(1, 5)]
        scores, fences, eggs = dict.fromkeys(names, 0), dict.fromkeys(names, 0), 0
        previous_finisher, game_result = None, None
        output = []
        for number, rnd in enumerate(rounds, 1):
            if game_result:
                raise InvalidGame(f"Кон {number}: партия уже завершена")
            if rnd.get("number") != number:
                raise InvalidGame("Нарушена нумерация конов")
            hands = deepcopy(rnd["deal"])
            self._check_deal(hands, order)
            starter = (
                next(p for p in order if "1-1" in hands[p])
                if not any(scores.values())
                else previous_finisher
            )
            turn, ends, last_non_double, finished = order.index(starter), None, None, None
            for index, move in enumerate(rnd["moves"], 1):
                where = f"Кон {number}, ход {index}"
                if finished:
                    raise InvalidGame(f"{where}: действие после завершения кона")
                player = move["player"]
                if player != order[turn]:
                    raise InvalidGame(f"{where}: нарушена очередь, ожидается {order[turn]}")
                action = move["action"]
                if index == 1:
                    if action != "start" or (not any(scores.values()) and move["stone"] != "1-1"):
                        raise InvalidGame(f"{where}: неверный заход")
                elif action == "start":
                    raise InvalidGame(f"{where}: повторный заход")
                if action == "pass":
                    if ends is None or self._can_play(hands[player], ends):
                        raise InvalidGame(f"{where}: необязательный пас")
                else:
                    try:
                        a, b = map(int, move["stone"].split("-"))
                    except (KeyError, ValueError):
                        raise InvalidGame(f"{where}: неверный камень") from None
                    stone = f"{min(a, b)}-{max(a, b)}"
                    if stone not in hands[player]:
                        raise InvalidGame(f"{where}: чужой или повторный камень {stone}")
                    if action == "start":
                        ends = [a, b]
                    elif action in ("left", "right"):
                        side = 0 if action == "left" else 1
                        if ends[side] not in (a, b):
                            raise InvalidGame(f"{where}: несовпадающий край")
                        ends[side] = b if ends[side] == a else a
                    else:
                        raise InvalidGame(f"{where}: неизвестное действие")
                    hands[player].remove(stone)
                    if a != b:
                        last_non_double = player
                    if not hands[player]:
                        finished = ("emptyHand", player, team_for[player])
                    elif not any(self._can_play(hand, ends) for hand in hands.values()):
                        finished = ("fish", last_non_double, None)
                turn = (turn + 1) % 4
            if not finished:
                raise InvalidGame(f"Кон {number}: розыгрыш не завершён")
            reason, finisher, winner = finished
            remaining = {
                team: sum(self._rules.hand_value(hands[p]) for p in order if team_for[p] == team)
                for team in scores
            }
            result = self._rules.settle(variant, reason, winner, remaining, scores, fences, eggs)
            result = {"reason": result.pop("reason"), "finished_player": finisher, **result}
            scores = result["total_score"]
            fences = {team: result.get("fence", {}).get(team, 0) for team in scores}
            eggs, previous_finisher = result.get("eggs", 0), finisher
            output.append(
                dict(
                    number=number,
                    deal=deepcopy(rnd["deal"]),
                    moves=deepcopy(rnd["moves"]),
                    result=result,
                )
            )
            game_result = self._rules.game_result(scores, limit)
        if game_result is None:
            raise InvalidGame("Партия не завершена")
        return dict(
            format="kozel-game-v1",
            fishVariant=variant,
            scoreLimit=limit,
            teams=deepcopy(teams),
            rounds=output,
            result=game_result,
        )

    def _can_play(self, hand, ends):
        return any(any(int(v) in ends for v in stone.split("-")) for stone in hand)

    def _check_deal(self, hands, players):
        if set(hands) != set(players) or any(len(hand) != 7 for hand in hands.values()):
            raise InvalidGame("Раздача должна содержать четыре руки по семь")
        all_stones = [stone for hand in hands.values() for stone in hand]
        expected = {f"{a}-{b}" for a in range(7) for b in range(a, 7)}
        if len(all_stones) != 28 or set(all_stones) != expected:
            raise InvalidGame("Раздача должна содержать все 28 различных камней")
        for hand in hands.values():
            if sum(stone[0] == stone[2] for stone in hand) >= 5:
                raise InvalidGame("Раздача требует пересдачи: пять дублей")
            if any(sum(str(pip) in stone for stone in hand) >= 6 for pip in range(7)):
                raise InvalidGame("Раздача требует пересдачи: шесть камней масти")
