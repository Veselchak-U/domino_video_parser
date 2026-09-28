"""Choose a recording timestamp without depending on filesystem dates."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import av


@dataclass(frozen=True)
class RecordingTime:
    prefix: str
    original: str
    source: str
    reason: str | None = None


class RecordingTimeResolver:
    def __init__(self, clock=None, to_local=None):
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._to_local = to_local or (lambda value: value.astimezone())

    def resolve(self, path: Path) -> RecordingTime:
        candidates = set()
        for match in re.finditer(r"(?<![0-9])[0-9]{4}(?:-[0-9]{2}){4}(?![0-9])", path.stem):
            value = match[0]
            try:
                datetime.strptime(value, "%Y-%m-%d-%H-%M")
            except ValueError:
                continue
            candidates.add(value)
        if len(candidates) > 1:
            raise ValueError("Неоднозначная дата в имени видео: " + ", ".join(sorted(candidates)))
        if candidates:
            value = candidates.pop()
            return RecordingTime(value, value, "filename")

        reason = "В имени и метаданных видео нет корректной даты-времени"
        try:
            with av.open(str(path)) as container:
                sources = [("container.creation_time", container.metadata.get("creation_time"))]
                if container.streams.video:
                    sources.append(
                        (
                            "video.creation_time",
                            container.streams.video[0].metadata.get("creation_time"),
                        )
                    )
                for source, original in sources:
                    parsed = self._parse_metadata(original)
                    if parsed is not None:
                        if parsed.tzinfo is not None:
                            parsed = self._to_local(parsed)
                        return RecordingTime(parsed.strftime("%Y-%m-%d-%H-%M"), original, source)
        except (OSError, av.FFmpegError, ValueError) as error:
            reason = f"Не удалось прочитать метаданные даты: {error}"

        now = self._to_local(self._clock())
        prefix = now.strftime("%Y-%m-%d-%H-%M-%S") + f"-{now.microsecond // 1000:03d}"
        return RecordingTime(prefix, now.isoformat(), "current_time", reason)

    def _parse_metadata(self, value):
        if not isinstance(value, str) or not re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ][0-9]{2}:[0-9]{2}"
            r"(?::[0-9]{2}(?:\.[0-9]+)?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?",
            value,
        ):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
