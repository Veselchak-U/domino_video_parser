"""Ограниченная по объёму трасса причин отклонения ветвей реконструкции."""

from copy import deepcopy


REASONS = {
    "unknown_player": "неизвестен исполнитель",
    "hand_capacity_exceeded": "переполнение руки",
    "initial_hand_conflict": "конфликт с начальной рукой",
    "prior_pass_conflict": "противоречие прежнему пасу",
    "illegal_pass": "невозможный пас",
    "geometry_conflict": "конфликт геометрии края",
    "endpoint_mismatch": "камень не подходит к концу",
    "hand_counter_conflict": "конфликт счётчика руки",
    "incomplete_deal": "восстановлена неполная раздача",
}


def event_trace(event, index, states):
    action = event.get("action")
    warnings = []
    if (index == 0 and action not in (None, "start")) or (index > 0 and action == "start"):
        warnings.append(dict(code="start_position_conflict", observed_action=action))
    return dict(
        event=index + 1,
        time=event.get("time"),
        stone=event.get("stone"),
        observed_action=action,
        candidate_seats=[None if s is None else s + 1 for s in event.get("seats", [event["seat"]])],
        states_before=states,
        states_after=0,
        warnings=warnings,
        rejections={},
    )


def reject(entry, code, **evidence):
    reason = entry["rejections"].setdefault(
        code, dict(count=0, examples=[], omitted_examples=0)
    )
    reason["count"] += 1
    if len(reason["examples"]) < 2:
        reason["examples"].append(deepcopy(evidence))
    else:
        reason["omitted_examples"] += 1


def failure_message(number, failure):
    message = f"Кон {number}: нет согласованной последовательности"
    if failure is not None:
        if failure["event"] is not None:
            message += f"; событие {failure['event']}, камень {failure['stone']}"
        if failure["time"] is not None:
            message += f", время {failure['time']:.3f} с"
        reasons = ", ".join(
            f"{REASONS[code]} ({reason['count']})"
            for code, reason in failure["rejections"].items()
        )
        message += f"; отклонённые ветви: {reasons}"
    return message
