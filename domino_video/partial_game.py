"""Структурный контракт неполной партии; игровую законность проверяет v1."""

import hashlib
import json
import math
from copy import deepcopy
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator


def _order_partial_events(
    events: list[dict[str, Any]], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    origin_order = {
        origin["event_id"]: index for index, origin in enumerate(raw.get("origin_events", []))
    }

    def source_key(event: dict[str, Any]) -> str | None:
        return event.get("parent_event_id", event.get("event_id"))

    identified = sorted(
        (event for event in events if source_key(event) in origin_order),
        key=lambda event: origin_order[source_key(event)],
    )
    ordered = iter(identified)
    timeline = sorted(events, key=lambda event: event.get("time", event.get("interval", [0])[0]))
    # Нижняя граница интервала служит внутренним якорем, а не новым доказательством
    # порядка. Существующие origins и supplied siblings сохраняют исходный порядок;
    # новые неподтверждённые промежутки остаются в позиции временного участка.
    return [next(ordered) if source_key(event) in origin_order else event for event in timeline]


def _partial_events(
    raw: dict[str, Any], diagnostics: list[dict[str, Any]], number: int
) -> list[dict[str, Any]]:
    events = deepcopy(raw.get("events", []))
    for unresolved in raw.get("unresolved", []):
        interval = unresolved.get("interval", [unresolved.get("time", 0)] * 2)
        values = [
            c.get("stone") if isinstance(c, dict) else c for c in unresolved.get("candidates", [])
        ]
        covered = any(
            e.get("stone") is not None
            and e.get("stone") in values
            and interval[0] <= e.get("time", interval[0]) <= interval[1]
            for e in events
        )
        duplicate = any(
            e.get("event_id") and e.get("event_id") == unresolved.get("event_id") for e in events
        )
        if not covered and not duplicate:
            events.append(deepcopy(unresolved))
    for reviewed in raw.get("reviewed_no_moves", []):
        events.append(dict(deepcopy(reviewed), seat=None, time=reviewed["interval"][0]))
    missing = set(raw.get("missing", []))
    remaining = raw.get("remaining")
    if remaining is not None and all(raw.get("remaining_confirmed", [False] * 4)):
        used = {e.get("stone") for e in events if e.get("stone") is not None}
        used.update(stone for hand in remaining for stone in hand)
        missing.update({f"{a}-{b}" for a in range(7) for b in range(a, 7)} - used)
    failures = [d for d in diagnostics if d.get("round") == number]
    failure = raw.get("first_failure") or next(
        (
            d.get("details", {}).get("first_failure")
            for d in failures
            if d.get("details", {}).get("first_failure")
        ),
        None,
    )
    for diagnostic in failures:
        missing.update(diagnostic.get("details", {}).get("missing", []))
    unknown = [
        e for e in events if e.get("stone") is None and e.get("review_resolution") != "no_move"
    ]
    unknown_gap = any(e.get("metadata", {}).get("unknown_count", 1) is None for e in unknown)
    known_times = [e.get("time", 0) for e in events if e.get("stone") is not None]
    final_interval = [max(known_times), raw.get("end")] if known_times else None
    uncovered_final = False
    if missing and unknown and final_interval and final_interval[1] is not None:

        def overlaps_final(event: dict[str, Any]) -> bool:
            interval = event.get("interval", [event.get("time", 0)] * 2)
            if isinstance(interval, dict):
                interval = [interval["start"], interval["end"]]
            return interval[0] <= final_interval[1] and interval[1] >= final_interval[0]

        uncovered_final = final_interval[0] <= final_interval[1] and not any(
            overlaps_final(event) for event in unknown
        )
    if uncovered_final or (
        not unknown_gap and ((missing and len(missing) > len(unknown)) or (failure and not unknown))
    ):
        interval = (
            final_interval
            if uncovered_final
            else (failure.get("interval") if isinstance(failure, dict) else None)
        )
        interval = interval or [
            max((e.get("time", 0) for e in events), default=raw.get("start") or 0),
            raw.get("end") or max((e.get("time", 0) for e in events), default=0),
        ]
        reviewed_gap = next(
            (
                marker
                for marker in raw.get("reviewed_no_moves", [])
                if marker["interval"] == interval
                and any(
                    origin["event_id"] == marker["event_id"]
                    and origin.get("reason") == "Неучтённые события или противоречие реконструкции"
                    and origin.get("metadata", {}).get("unknown_count", 1) is None
                    for origin in raw.get("origin_events", [])
                )
            ),
            None,
        )
        if reviewed_gap and not missing:
            return _order_partial_events(events, raw)
        gap_identity = {}
        if reviewed_gap:
            identity = json.dumps([reviewed_gap["event_id"], sorted(missing)], sort_keys=True)
            gap_identity["event_id"] = "event-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        events.append(
            dict(
                **gap_identity,
                seat=None,
                action=None,
                stone=None,
                interval=interval,
                time=interval[0],
                time_is_anchor=True,
                candidates=[{"stone": stone} for stone in sorted(missing)],
                reason="Неучтённые события или противоречие реконструкции",
                metadata=dict(unknown_count=None, order_candidates=[]),
            )
        )
    return _order_partial_events(events, raw)


def build_partial_game(
    rounds: list[dict[str, Any]],
    names: list[str] | None,
    variant: str,
    limit: int,
    sha256: str,
    game_number: int,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    names = names or [f"Unrecognized_{i + 1}" for i in range(4)]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("Нужны четыре разных имени")
    teams = [
        dict(id=key, name=f"Team {key}", players=[dict(name=names[i], seat=i + 1) for i in seats])
        for key, seats in (("A", (0, 2)), ("B", (1, 3)))
    ]
    output = []
    for number, raw in enumerate(rounds, 1):
        moves = []
        for event in _partial_events(raw, diagnostics or [], number):
            interval = event.get("interval")
            if isinstance(interval, dict):
                interval = [interval["start"], interval["end"]]
            if interval is None:
                interval = [event.get("time", raw.get("start", 0))] * 2
            anchors = deepcopy(event.get("anchors", {}))
            identity = json.dumps(
                [
                    sha256,
                    game_number,
                    number,
                    interval,
                    anchors,
                    event.get("seat"),
                    event.get("action"),
                    event.get("stone"),
                    event.get("positions"),
                    event.get("placement_links"),
                ],
                sort_keys=True,
            )
            event_id = (
                event.get("event_id")
                or "event-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            )
            seat = event.get("seat")
            action = event.get("action") or "unknown"
            stone = event.get("orientation") or event.get("stone")
            candidates = deepcopy(event.get("candidates", []))
            candidates = [
                {"stone": value} if isinstance(value, str) else value for value in candidates
            ]
            if (
                stone is not None
                and action in ("left", "right", "unknown")
                and not event.get("orientation")
            ):
                if not candidates:
                    candidates = [{"stone": stone}]
                stone = None
            known = (
                seat is not None and action != "unknown" and (action == "pass" or stone is not None)
            )
            move = dict(
                event_id=event_id,
                recognition_status=event.get(
                    "recognition_status", "confirmed" if known else "unresolved"
                ),
                player=names[seat] if seat is not None else None,
                action=action,
                interval=list(interval),
                reason=None if known else event.get("reason", "Неоднозначное событие"),
                candidates=candidates,
            )
            if action != "pass":
                move["stone"] = stone
            for field in ("metadata", "parent_event_id", "anchors", "review_resolution"):
                if field in event:
                    move[field] = deepcopy(event[field])
            moves.append(move)
        for marker in raw.get("supplied_passes", []):
            index = next(
                (i for i, move in enumerate(moves) if move["event_id"] == marker["after_event_id"]),
                len(moves),
            )
            moves.insert(index, deepcopy(marker["move"]))
        deal = dict.fromkeys(names)
        hand = raw.get("initial_hand")
        if hand is not None and len(hand) == 7 and len(set(hand)) == 7:
            deal[names[0]] = list(hand)
        for seat, hand in raw.get("deal", {}).items():
            if type(seat) is int and seat in range(4) and hand is not None:
                if len(hand) == 7 and len(set(hand)) == 7:
                    deal[names[seat]] = list(hand)
        evidence = {
            key: deepcopy(raw[key])
            for key in ("score", "remaining", "remaining_confirmed", "initial_hand")
            if key in raw
        }
        origins = deepcopy(raw.get("origin_events", moves))
        origin_ids = {move["event_id"] for move in origins}
        for index, move in enumerate(moves):
            if move["event_id"] in origin_ids or move.get("parent_event_id"):
                continue
            next_id = next(
                (
                    later.get("parent_event_id", later["event_id"])
                    for later in moves[index + 1 :]
                    if later.get("parent_event_id", later["event_id"]) in origin_ids
                ),
                None,
            )
            position = next(
                (i for i, origin in enumerate(origins) if origin["event_id"] == next_id),
                len(origins),
            )
            origins.insert(position, deepcopy(move))
            origin_ids.add(move["event_id"])
        output.append(
            dict(
                number=number,
                deal=deal,
                moves=moves,
                origin_events=origins,
                result=None,
                evidence=evidence,
            )
        )
    document = dict(
        format="kozel-game-v2",
        status="needs_review",
        source=dict(sha256=sha256, game_number=game_number),
        fishVariant=variant,
        scoreLimit=limit,
        teams=teams,
        rounds=output,
        result=None,
        diagnostics=deepcopy(diagnostics or []),
    )
    validate_partial_game(document)
    return document


def validate_partial_game(document: dict[str, Any]) -> None:
    schema = json.loads(
        files("domino_video").joinpath("schema_v2.json").read_text(encoding="utf-8")
    )
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        raise ValueError("Неверная структура v2: " + errors[0].message)
    names = [p["name"] for team in document["teams"] for p in team["players"]]
    seats = [p["seat"] for team in document["teams"] for p in team["players"]]
    if (
        len(set(names)) != 4
        or set(seats) != {1, 2, 3, 4}
        or len({t["id"] for t in document["teams"]}) != 2
    ):
        raise ValueError("Повтор имени, места или команды v2")
    for team in document["teams"]:
        if abs(team["players"][0]["seat"] - team["players"][1]["seat"]) != 2:
            raise ValueError("Команда должна сидеть напротив")
    for number, rnd in enumerate(document["rounds"], 1):
        if rnd["number"] != number or set(rnd["deal"]) != set(names):
            raise ValueError("Неверный номер кона или игрок раздачи")
        origins = {m["event_id"]: m for m in rnd["origin_events"]}
        if len(origins) != len(rnd["origin_events"]) or len(
            {m["event_id"] for m in rnd["moves"]}
        ) != len(rnd["moves"]):
            raise ValueError("Повтор event_id")
        referenced = set()
        origin_order = {key: index for index, key in enumerate(origins)}
        previous = -1
        replaced = set()
        for move in rnd["origin_events"] + rnd["moves"]:
            start, end = move["interval"]
            if not math.isfinite(start) or not math.isfinite(end) or start > end:
                raise ValueError("Неверный интервал события")
            if move["player"] is not None and move["player"] not in names:
                raise ValueError("Неизвестный игрок события")
            if move["recognition_status"] == "confirmed" and (
                move["player"] is None
                or move["action"] == "unknown"
                or (move["action"] != "pass" and move["stone"] is None)
            ):
                raise ValueError("Подтверждённое событие содержит неизвестные поля")
        for move in rnd["moves"]:
            parent = move.get("parent_event_id")
            key = parent or move["event_id"]
            if key not in origins:
                raise ValueError("Неизвестная ссылка event_id/parent_event_id")
            origin = origins[key]
            if (
                move.get("review_resolution") == "no_move"
                and origin["recognition_status"] != "unresolved"
            ):
                raise ValueError("no_move разрешён только для исходного unresolved-события")
            if origin_order[key] < previous:
                raise ValueError("Порядок исходных событий изменён")
            previous = origin_order[key]
            if parent and move["event_id"] in origins:
                raise ValueError("Новое событие использует исходный event_id")
            if key in replaced and not parent:
                raise ValueError("Промежуток одновременно сохранён и заменён")
            if parent:
                if key in referenced and key not in replaced:
                    raise ValueError("Промежуток одновременно сохранён и заменён")
                replaced.add(key)
            referenced.add(key)
            if parent and (
                origin["recognition_status"] != "unresolved"
                or "metadata" not in origin
                or move["recognition_status"] != "supplied"
            ):
                raise ValueError("Заменять последовательностью можно только unresolved-промежуток")
            if (
                move["interval"][0] < origin["interval"][0]
                or move["interval"][1] > origin["interval"][1]
                or (not parent and move["interval"] != origin["interval"])
                or move.get("anchors") != origin.get("anchors")
            ):
                raise ValueError("Интервал или якоря несовместимы с исходным событием")
            if (
                any(move.get(k) != origin.get(k) for k in ("player", "action", "stone"))
                and move["recognition_status"] != "supplied"
            ):
                raise ValueError("Изменённые поля требуют recognition_status=supplied")
        if referenced != set(origins):
            raise ValueError("Исходное событие удалено без замены")
