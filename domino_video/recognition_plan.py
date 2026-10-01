"""Select text fields from visual evidence before running expensive OCR."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vision import Observation


@dataclass(frozen=True)
class OCRFields:
    names: bool = False
    counts: bool = False
    scores: bool = False


def recognition_requests(
    rounds: list[dict], observations: list["Observation"]
) -> dict[float, OCRFields]:
    requests = {}

    def add(time, field):
        requests[time] = replace(requests.get(time, OCRFields()), **{field: True})

    for i, rnd in enumerate(rounds):
        if not rnd.get("complete"):
            continue
        end = rnd["end"]
        stop = rounds[i + 1]["start"] if i + 1 < len(rounds) else float("inf")
        after = [o for o in observations if end < o.time < stop]
        tables = [o for o in after if o.result_table]
        for obs in tables[:3]:
            add(obs.time, "names")
        score_stop = stop if i + 1 < len(rounds) else end + 16
        scores = [
            o
            for o in after
            if o.supported
            and not o.reveal
            and not o.board
            and not o.result_table
            and o.time < score_stop
        ]
        # Late readings follow the score animation rather than its old value.
        for obs in scores[-3:]:
            add(obs.time, "scores")
        events = rnd["events"]
        for n, event in enumerate(events):
            if not rnd.get("indicator_unreliable") and len(set(event.get("seats", []))) <= 1:
                continue
            until = events[n + 1]["time"] if n + 1 < len(events) else end
            rows = [
                o
                for o in observations
                if event["time"] + 0.3 < o.time < until - 0.3
                and o.supported
                and not o.reveal
                and not o.result_table
            ]
            for obs in rows[:2]:
                add(obs.time, "counts")
    return requests
