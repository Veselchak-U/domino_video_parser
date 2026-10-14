"""Independent visual evidence for empty final hands."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vision import Observation


def terminal_empty_seat(rnd: dict) -> int | None:
    """Select an already confirmed empty hand needing independent terminal proof."""
    hands = rnd.get("remaining")
    confirmed = rnd.get("remaining_confirmed", [])
    if (
        not rnd.get("complete")
        or rnd.get("end") is None
        or not hands
        or len(hands) != 4
        or len(confirmed) != 4
        or not all(confirmed)
        or sum(not hand for hand in hands) != 1
    ):
        return None
    known = [event.get("stone") for event in rnd.get("events", [])]
    known += [stone for hand in hands for stone in hand]
    deck = {f"{a}-{b}" for a in range(7) for b in range(a, 7)}
    if len(known) != 27 or len(set(known)) != 27 or not set(known) <= deck:
        return None
    seat = next(index for index, hand in enumerate(hands) if not hand)
    if any(
        proof.get("method") == "zero_reveal_points" and proof.get("seat") == seat + 1
        for proof in rnd.get("remaining_evidence", [])
    ):
        return None
    return seat


def confirm_empty_remaining(rounds: list[dict], observations: list["Observation"]) -> None:
    for index, rnd in enumerate(rounds):
        if not rnd.get("complete") or rnd.get("end") is None:
            continue
        stop = rounds[index + 1]["start"] if index + 1 < len(rounds) else float("inf")
        rows = [o for o in observations if rnd["end"] <= o.time < stop and o.reveal]
        confirmed = rnd.get("remaining_confirmed", [False] * 4)
        terminal_seat = terminal_empty_seat(rnd)
        for seat, hand in enumerate(rnd.get("remaining") or []):
            if (
                (confirmed[seat] and seat != terminal_seat)
                or hand
                or any(o.hands[seat] for o in rows)
            ):
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
                    and (
                        seat != terminal_seat
                        or (o.reveal_panels_complete and o.reveal_panels_complete[seat])
                    )
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
