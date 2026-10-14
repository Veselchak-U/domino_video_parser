import math
from copy import deepcopy

from .partial_game import _partial_events, build_partial_game, validate_partial_game


class Corrections:
    def __init__(self, document=None):
        self._sources = {}
        self._partial = None
        if document is None:
            return
        if isinstance(document, dict) and document.get("format") == "kozel-game-v2":
            validate_partial_game(document)
            self._partial = deepcopy(document)
            source = document["source"]
            self._sources[source["sha256"]] = {source["game_number"]: {}}
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

    def check_settings(self, variant: str, limit: int) -> None:
        if self._partial is not None and (
            self._partial["fishVariant"] != variant or self._partial["scoreLimit"] != limit
        ):
            raise ValueError("Вариант правил или лимит v2 не совпадает с параметрами запуска")

    def check_games(self, sha256, count):
        if any(n > count for n in self._sources.get(sha256, {})):
            raise ValueError("Исправления ссылаются на отсутствующую партию")

    def apply(self, sha256, game_number, rounds):
        if self._partial is not None:
            if sha256 != self._partial["source"]["sha256"]:
                return deepcopy(rounds), None
            if game_number == self._partial["source"]["game_number"]:
                return self._apply_partial(rounds)
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
                rnd.setdefault("remaining_confirmed", [False] * 4)[int(seat)] = True
                if seat == "0":
                    rnd["initial_hand"] = list(hand)
        return result, game.get("teams")

    def _apply_partial(self, rounds: list[dict]) -> tuple[list[dict], list[dict]]:
        document = self._partial
        names = [None] * 4
        for team in document["teams"]:
            for player in team["players"]:
                names[player["seat"] - 1] = player["name"]
        baseline = build_partial_game(
            rounds,
            names,
            document["fishVariant"],
            document["scoreLimit"],
            document["source"]["sha256"],
            document["source"]["game_number"],
            document.get("diagnostics", []),
        )
        if len(document["rounds"]) != len(rounds):
            raise ValueError("Исправления ссылаются на отсутствующий кон")
        result = deepcopy(rounds)
        for raw, supplied, current in zip(
            result, document["rounds"], baseline["rounds"], strict=True
        ):
            current_origins = {move["event_id"]: move for move in current["origin_events"]}
            for origin in supplied["origin_events"]:
                actual = current_origins.get(origin["event_id"])
                if actual is None or any(
                    origin.get(k) != actual.get(k)
                    for k in (
                        "interval",
                        "anchors",
                        "player",
                        "action",
                        "stone",
                        "recognition_status",
                        "metadata",
                    )
                ):
                    raise ValueError("Исходная ссылка или якоря исправления не совпадают с видео")
            if {m["event_id"] for m in supplied["origin_events"]} != set(current_origins):
                raise ValueError("Исправления потеряли исходное событие")
            partial_events = _partial_events(
                raw, document.get("diagnostics", []), supplied["number"]
            )
            by_id = {
                move["event_id"]: event
                for move, event in zip(
                    [m for m in current["moves"] if m["action"] != "pass"],
                    partial_events,
                    strict=True,
                )
            }
            events = []
            supplied_passes = []
            reviewed_no_moves = []
            for move_index, move in enumerate(supplied["moves"]):
                if move.get("review_resolution") == "no_move":
                    reviewed_no_moves.append(deepcopy(move))
                    continue
                event = deepcopy(by_id[move.get("parent_event_id", move["event_id"])])
                if move["recognition_status"] == "supplied":
                    event.pop("seats", None)
                    event["supplied"] = True
                    event.update(
                        seat=names.index(move["player"]) if move["player"] is not None else None,
                        action=None if move["action"] == "unknown" else move["action"],
                        stone=move.get("stone"),
                        recognition_status="supplied",
                    )
                    if move.get("stone") is not None:
                        event["orientation"] = move["stone"]
                        event["stone"] = "-".join(sorted(move["stone"].split("-")))
                    else:
                        event.pop("orientation", None)
                exact_time = (
                    event.get("time")
                    if not event.get("time_is_anchor")
                    and move["interval"][0] == move["interval"][1]
                    else None
                )
                event.update(
                    event_id=move["event_id"],
                    interval=deepcopy(move["interval"]),
                    time=exact_time if exact_time is not None else move["interval"][0],
                    time_is_anchor=exact_time is None,
                )
                if "parent_event_id" in move:
                    event["parent_event_id"] = move["parent_event_id"]
                if move["action"] == "pass" and move["recognition_status"] == "supplied":
                    before = next(
                        (
                            item["event_id"]
                            for item in reversed(supplied["moves"][:move_index])
                            if item["action"] != "pass"
                            and item.get("review_resolution") != "no_move"
                        ),
                        None,
                    )
                    after = next(
                        (
                            item["event_id"]
                            for item in supplied["moves"][move_index + 1 :]
                            if item["action"] != "pass"
                            and item.get("review_resolution") != "no_move"
                        ),
                        None,
                    )
                    supplied_passes.append(
                        dict(
                            time=event["time"],
                            interval=event["interval"],
                            seat=event["seat"],
                            before_event_id=before,
                            after_event_id=after,
                            move=deepcopy(move),
                        )
                    )
                else:
                    events.append(event)
            raw["events"] = events
            raw["unresolved"] = []
            raw["supplied_passes"] = supplied_passes
            raw["reviewed_no_moves"] = reviewed_no_moves
            raw["origin_events"] = deepcopy(supplied["origin_events"])
            for seat, name in enumerate(names):
                hand = supplied["deal"][name]
                if hand is None:
                    continue
                rest = list(hand)
                for event in events:
                    if event.get("seat") == seat and event.get("stone") is not None:
                        if event["stone"] not in rest:
                            raise ValueError("Ход отсутствует в исправленной руке")
                        rest.remove(event["stone"])
                if all(
                    event.get("stone") is not None and event.get("seat") is not None
                    for event in events
                ):
                    raw.setdefault("remaining", [[], [], [], []])[seat] = rest
                    raw.setdefault("remaining_confirmed", [False] * 4)[seat] = True
                if seat == 0:
                    raw["initial_hand"] = list(hand)
                raw.setdefault("deal", {})[seat] = list(hand)
        return result, deepcopy(document["teams"])

    def _keys(self, obj, allowed):
        if not isinstance(obj, dict) or set(obj) - allowed:
            raise ValueError("Неизвестные поля исправлений")
