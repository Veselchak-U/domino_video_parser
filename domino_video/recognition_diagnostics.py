"""Select evidence for final OCR failures, without decoding or writing files."""

from collections import Counter
from copy import deepcopy

from .ocr_result import OCR_THRESHOLD
from .reconstruct import NameRecognitionError, ScoreRecognitionError
from .screen_profile import ScreenProfile


class RecognitionDiagnostics:
    reasons = {
        "low_confidence": "низкая уверенность OCR",
        "no_text": "OCR не обнаружил текст",
        "invalid_format": "неверный формат текста",
        "duplicate_name": "повтор имени",
        "no_frame": "нет пригодного кадра для OCR",
    }
    fields = {"name": "имя", "count": "счётчик камней", "score": "итоговый счёт"}

    def build(self, observations, group, game_number, error=None):
        start, end = group[0]["start"] - 10, group[-1]["end"] or float("inf")
        in_game = sorted((o for o in observations if start <= o.time <= end), key=lambda o: o.time)
        entries = []
        if isinstance(error, NameRecognitionError):
            readings = [[a for a in o.ocr_attempts if a["field"] == "name"] for o in in_game]
            readings = [r for r in readings if len(r) == 4]
            if readings:
                best = max(
                    readings,
                    key=lambda r: len({a["accepted_text"] for a in r if a["accepted_text"]}),
                )
                duplicates = Counter(a["accepted_text"] for a in best)
                for attempt in best:
                    value = attempt["accepted_text"]
                    if not value or duplicates[value] > 1:
                        entry = self._entry(attempt, game_number)
                        if value:
                            entry["reason"] = "duplicate_name"
                        entries.append(entry)
            else:
                entries.extend(self._missing("name", game_number, seat=i) for i in range(1, 5))

        for seat in range(2, 5):
            attempts = [
                a
                for o in in_game
                for a in o.ocr_attempts
                if a["field"] == "count" and a["seat"] == seat
            ]
            if attempts and all(a["reason"] for a in attempts):
                entries.append(self._entry(attempts[0], game_number, severity="warning"))

        if isinstance(error, ScoreRecognitionError):
            for number, rnd in enumerate(group, 1):
                stop = group[number]["start"] if number < len(group) else rnd["end"] + 16
                rows = sorted(
                    (o for o in observations if rnd["end"] < o.time < stop), key=lambda o: o.time
                )
                if any(o.scores for o in rows):
                    continue
                for team in "AB":
                    attempts = [
                        a
                        for o in rows
                        for a in o.ocr_attempts
                        if a["field"] == "score" and a["team"] == team
                    ]
                    failed = [a for a in attempts if a["reason"]]
                    if failed:
                        entries.append(self._entry(failed[0], game_number, number))
                    elif not attempts:
                        entries.append(self._missing("score", game_number, number, team=team))
        return entries

    def _entry(self, attempt, game, round_number=None, severity="error"):
        entry = deepcopy(attempt)
        entry.update(game=game, round=round_number, severity=severity, sample=None)
        entry["position"] = ScreenProfile.positions[entry["seat"] - 1] if entry["seat"] else None
        return entry

    def _missing(self, field, game, round_number=None, seat=None, team=None):
        return self._entry(
            dict(
                field=field,
                seat=seat,
                team=team,
                time=None,
                raw_rows=[],
                accepted_text="",
                threshold=OCR_THRESHOLD,
                reason="no_frame",
            ),
            game,
            round_number,
        )

    def message(self, entry, report_dir):
        subject = (
            f"место {entry['seat']} ({entry['position']})"
            if entry["seat"]
            else f"команда {entry['team']}"
        )
        time = f"{entry['time']:.2f} сек" if entry["time"] is not None else "таймкод отсутствует"
        sample = entry["sample"]
        attachment = (
            f"образец: {report_dir / sample['path']}"
            if sample
            else f"образец отсутствует: {entry.get('sample_error', self.reasons['no_frame'])}"
        )
        prefix = f"Партия {entry['game']}"
        if entry["round"]:
            prefix += f", кон {entry['round']}"
        evidence = ", ".join(
            f"«{row['text']}» ({row['confidence']:.1%})" for row in entry["raw_rows"]
        )
        details = f"; OCR: {evidence}, порог > {entry['threshold']:.0%}" if evidence else ""
        return (
            f"{prefix}: {self.fields[entry['field']]}, {subject}, {time} — "
            f"{self.reasons[entry['reason']]}{details}; {attachment}"
        )
