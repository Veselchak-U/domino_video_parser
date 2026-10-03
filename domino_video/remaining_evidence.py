"""Independent visual evidence for empty final hands."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vision import Observation


def confirm_empty_remaining(rounds: list[dict], observations: list["Observation"]) -> None:
    for index, rnd in enumerate(rounds):
        if not rnd.get("complete") or rnd.get("end") is None:
            continue
        stop = rounds[index + 1]["start"] if index + 1 < len(rounds) else float("inf")
        rows = [o for o in observations if rnd["end"] <= o.time < stop and o.reveal]
        confirmed = rnd.get("remaining_confirmed", [False] * 4)
        for seat, hand in enumerate(rnd.get("remaining") or []):
            if confirmed[seat] or hand or any(o.hands[seat] for o in rows):
                continue
            points = [o.reveal_points[seat] for o in rows if o.reveal_points is not None]
            if any(value not in (None, 0) for value in points):
                continue
            times = sorted(
                {
                    o.time
                    for o in rows
                    if o.supported
                    and o.reveal_valid
                    and o.reveal_valid[seat]
                    and o.reveal_points is not None
                    and o.reveal_points[seat] == 0
                }
            )
            if len(times) < 2:
                continue
            confirmed[seat] = True
            rnd["remaining_confirmed"] = confirmed
            rnd.setdefault("remaining_evidence", []).append(
                dict(
                    seat=seat + 1, method="zero_reveal_points", times=times, points=[0] * len(times)
                )
            )
