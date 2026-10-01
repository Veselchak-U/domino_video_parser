"""Choose stable, unique player names from evidence belonging to one game."""

from copy import deepcopy
from dataclasses import dataclass

from .ocr_result import NAME_OCR_THRESHOLD
from .screen_profile import ScreenProfile


@dataclass(frozen=True)
class ResolvedNames:
    names: list[str]
    entries: list[dict]


class PlayerNameResolver:
    def resolve(self, observations, group):
        start, end = group[0]["start"] - 10, group[-1]["end"] or float("inf")
        if "names_end" in group[-1]:
            start, end = group[-1]["end"], group[-1]["names_end"]
        choices = []
        for seat in range(1, 5):
            candidates = []
            for obs in observations:
                if not start <= obs.time <= end:
                    continue
                attempts = [
                    a for a in obs.ocr_attempts if a["field"] == "name" and a["seat"] == seat
                ]
                if not attempts and obs.names and len(obs.names) == 4:
                    # Already accepted observations (including stored timelines) have
                    # no individual probabilities. Do not invent OCR evidence.
                    attempts = [
                        dict(
                            field="name",
                            seat=seat,
                            team=None,
                            time=obs.time,
                            accepted_text=obs.names[seat - 1],
                            raw_rows=[],
                            reason=None,
                            threshold=NAME_OCR_THRESHOLD,
                        )
                    ]
                for a in attempts:
                    value, masked, confident = self._reading(a)
                    probabilities = [
                        s["confidence"]
                        for s in a.get("symbols", [])
                        if s["confidence"] > NAME_OCR_THRESHOLD
                    ]
                    quality = (
                        len(confident),
                        -len(masked),
                        sum(probabilities) / len(probabilities) if probabilities else 0,
                        -a["time"],
                    )
                    if "symbols" not in a:
                        quality = (sum(not c.isspace() for c in value), 0, 0, -a["time"])
                    candidates.append((quality, value, masked, a))
            choices.append(max(candidates, key=lambda c: c[0]) if candidates else None)

        reserved = {c[1] for c in choices if c and c[1]}
        used, names, entries = set(), [None] * 4, []
        for i, choice in enumerate(choices):
            if not choice or not choice[1]:
                continue
            _, value, masked, a = choice
            name = value
            if name in used:
                suffix = 2
                while f"{value}_{suffix}" in reserved | used:
                    suffix += 1
                name = f"{value}_{suffix}"
            names[i] = name
            used.add(name)
            reasons = (["partial_name"] if masked else []) + (
                ["duplicate_name"] if name != value else []
            )
            if reasons:
                entries.append(self._entry(a, i + 1, name, masked, reasons))
        for i, choice in enumerate(choices):
            if names[i] is not None:
                continue
            number = 1
            while f"Unrecognized_{number}" in reserved | used:
                number += 1
            name = names[i] = f"Unrecognized_{number}"
            used.add(name)
            a = choice[3] if choice else None
            reasons = ["generated_name"] if a else ["no_frame", "generated_name"]
            entries.append(self._entry(a, i + 1, name, choice[2] if choice else [], reasons))
        return ResolvedNames(names, sorted(entries, key=lambda e: e["seat"]))

    def _reading(self, attempt):
        if "symbols" not in attempt:
            value = attempt["accepted_text"]
            return value if value.strip() else "", [], []
        symbols = attempt["symbols"]
        confident = [
            s["confidence"]
            for s in symbols
            if s["confidence"] > NAME_OCR_THRESHOLD and not s["text"].isspace()
        ]
        masked = [i for i, s in enumerate(symbols) if s["confidence"] <= NAME_OCR_THRESHOLD]
        text = "".join(s["text"] if s["confidence"] > NAME_OCR_THRESHOLD else "*" for s in symbols)
        return text if confident else "", masked, confident

    def _entry(self, attempt, seat, name, masked, reasons):
        entry = (
            deepcopy(attempt)
            if attempt
            else dict(
                field="name",
                seat=seat,
                team=None,
                time=None,
                raw_rows=[],
                symbols=[],
                accepted_text="",
                threshold=NAME_OCR_THRESHOLD,
            )
        )
        entry.update(
            final_name=name,
            masked_positions=masked,
            reasons=reasons,
            reason=reasons[0],
            position=ScreenProfile.positions[seat - 1],
        )
        return entry
