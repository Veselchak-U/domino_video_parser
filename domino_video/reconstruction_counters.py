"""Counter evidence for a reconstructed sequence of placements."""

from collections import Counter


def _ahead_player(counts, expected, next_event):
    if (
        next_event is None
        or len(counts) != 4
        or not all(type(value) is int and 0 <= value <= 7 for value in counts)
        or sum(counts) != sum(expected) - 1
    ):
        return None
    differences = [p for p, value in enumerate(counts) if value != expected[p]]
    eligible = next_event.get("seats", [next_event.get("seat")])
    if (
        len(differences) == 1
        and differences[0] in eligible
        and counts[differences[0]] == expected[differences[0]] - 1
    ):
        return differences[0]
    return None


def ahead_selection_counts(
    counters: list[dict], played: list[list[str]], next_event: dict | None
) -> Counter[int]:
    """Keep the next-player constraint that justified aligned counter values."""
    expected = [7 - len(hand) for hand in played]
    result = Counter()
    for observation in counters:
        seat = _ahead_player(observation["counts"], expected, next_event)
        if seat is not None:
            result[seat] += 1
    return result


def count_readings(
    counters: list[dict], played: list[list[str]], next_event: dict | None
) -> list[Counter[int]]:
    """Align only a fully explained selection of the next stone with this state.

    The UI can subtract a selected stone before its flight becomes readable.
    Four hands initially contain 28 stones. Therefore a complete counter vector
    can be one placement ahead only when exactly one eligible next player's
    count is lower by one and every other count equals the current reconstructed
    hand size. This is not a general tolerance for OCR errors: all unexplained
    vectors retain their original values, including partial and final readings.
    The caller still checks repetition and rejects conflicting branch evidence.
    """
    expected = [7 - len(hand) for hand in played]
    readings = [Counter() for _ in range(4)]
    for observation in counters:
        counts = observation["counts"]
        if _ahead_player(counts, expected, next_event) is not None:
            counts = expected
        for p, value in enumerate(counts[:4]):
            if value is not None:
                readings[p][value] += 1
    return readings
