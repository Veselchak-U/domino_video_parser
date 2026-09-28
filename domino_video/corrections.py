import math
from copy import deepcopy


class Corrections:
    def __init__(self, document=None):
        self._sources = {}
        if document is None:
            return
        self._keys(document, {"sources"})
        if not isinstance(document.get("sources"), list):
            raise ValueError("Исправления должны содержать массив sources")
        for source in document["sources"]:
            self._keys(source, {"sha256", "games"})
            key = source["sha256"]
            if key in self._sources or not isinstance(key, str):
                raise ValueError("Повтор или неверный хеш исправлений")
            games = {}
            for game in source["games"]:
                self._keys(game, {"number", "teams", "rounds"})
                n = game["number"]
                if type(n) is not int or n < 1 or n in games:
                    raise ValueError("Неверный номер партии исправлений")
                games[n] = game
            self._sources[key] = games

    def check_sources(self, hashes):
        if set(self._sources) - set(hashes):
            raise ValueError("Хеш исправлений не совпадает ни с одним входным видео")

    def check_games(self, sha256, count):
        if any(n > count for n in self._sources.get(sha256, {})):
            raise ValueError("Исправления ссылаются на отсутствующую партию")

    def apply(self, sha256, game_number, rounds):
        result = deepcopy(rounds)
        game = self._sources.get(sha256, {}).get(game_number, {})
        seen = set()
        for correction in game.get("rounds", []):
            self._keys(correction, {"number", "deal", "events"})
            number = correction["number"]
            if type(number) is not int or number < 1 or number > len(result) or number in seen:
                raise ValueError("Неверный или повторный номер кона исправлений")
            seen.add(number)
            rnd = result[number - 1]
            operations = {}
            for operation in correction.get("events", []):
                self._keys(operation, {"op", "index", "value"})
                index = operation["index"]
                op = operation["op"]
                if op not in ("replace", "insert", "delete") or type(index) is not int:
                    raise ValueError("Неверная операция исправления")
                maximum = len(rnd["events"]) + (op == "insert")
                if index < 1 or index > maximum or index in operations:
                    raise ValueError("Конфликт или отсутствующий индекс события")
                operations[index] = operation
            events = []
            for i in range(1, len(rnd["events"]) + 2):
                original = rnd["events"][i - 1] if i <= len(rnd["events"]) else None
                operation = operations.get(i)
                if operation and operation["op"] != "delete":
                    value = deepcopy(operation["value"])
                    self._keys(value, {"seat", "action", "stone", "time"})
                    if (
                        set(value) != {"seat", "action", "stone", "time"}
                        or type(value["seat"]) is not int
                        or value["seat"] not in range(4)
                    ):
                        raise ValueError("Неверный исполнитель или поля исправленного события")
                    if value["action"] not in ("start", "left", "right", None):
                        raise ValueError("Исправляются выкладки; пасы проверяются по рукам")
                    if (
                        type(value["time"]) not in (int, float)
                        or not math.isfinite(value["time"])
                        or value["time"] < 0
                    ):
                        raise ValueError("Таймкод должен быть конечным неотрицательным числом")
                    stones = {f"{a}-{b}" for a in range(7) for b in range(a, 7)}
                    if value["action"] == "start":
                        stones = {f"{a}-{b}" for a in range(7) for b in range(7)}
                    if value["stone"] not in stones:
                        raise ValueError("Неверный камень исправления")
                    if value["action"] == "start":
                        value["orientation"] = value["stone"]
                        value["stone"] = "-".join(sorted(value["stone"].split("-")))
                    events.append(value)
                if original is not None and (not operation or operation["op"] == "insert"):
                    events.append(original)
            rnd["events"] = events
            for seat, hand in correction.get("deal", {}).items():
                if (
                    seat not in ("0", "1", "2", "3")
                    or not isinstance(hand, list)
                    or len(hand) != 7
                    or len(set(hand)) != 7
                ):
                    raise ValueError(
                        "Исправленная рука должна содержать семь разных камней, место 0–3"
                    )
                rest = list(hand)
                for event in events:
                    if event["seat"] == int(seat):
                        if event["stone"] not in rest:
                            raise ValueError("Ход отсутствует в исправленной руке")
                        rest.remove(event["stone"])
                rnd["remaining"][int(seat)] = rest
                if seat == "0":
                    rnd["initial_hand"] = list(hand)
        return result, game.get("teams")

    def _keys(self, obj, allowed):
        if not isinstance(obj, dict) or set(obj) - allowed:
            raise ValueError("Неизвестные поля исправлений")
