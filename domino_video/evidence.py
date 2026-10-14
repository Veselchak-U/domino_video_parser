"""Evidence owned by one source parse; decoded images have a byte-bounded LRU."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from math import isfinite

import numpy as np

from .vision import Observation


class SourceEvidence:
    def __init__(
        self,
        source_sha256: str,
        stream: int = 0,
        profile: str = "1608x720",
        recognition_version: str = "1",
        image_budget_bytes: int = 64 * 1024 * 1024,
    ):
        if image_budget_bytes < 0:
            raise ValueError("Бюджет изображений не может быть отрицательным")
        self.identity = (source_sha256, stream, profile, recognition_version)
        self.image_budget_bytes = image_budget_bytes
        self.image_bytes = 0
        self._images = OrderedDict()
        self._observations = {}
        self._analyses = set()
        self._fields = {}
        self.conflicts = []
        self._closed = False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Контекст evidence закрыт")

    def close(self):
        self._images.clear()
        self._observations.clear()
        self._analyses.clear()
        self._fields.clear()
        self.conflicts.clear()
        self.image_bytes = 0
        self._closed = True

    @property
    def observations(self) -> list[Observation]:
        self._check_open()
        return [self._observations[t] for t in sorted(self._observations)]

    def cache_image(self, time: float, image: np.ndarray) -> None:
        self._check_open()
        if not isfinite(time):
            raise ValueError("Некорректный PTS")
        old = self._images.pop(time, None)
        if old is not None:
            self.image_bytes -= old.nbytes
        if image.nbytes > self.image_budget_bytes:
            return
        while self.image_bytes + image.nbytes > self.image_budget_bytes:
            _, old = self._images.popitem(last=False)
            self.image_bytes -= old.nbytes
        # Copy so caller-owned views cannot retain a larger backing allocation.
        stored = image.copy()
        self._images[time] = stored
        self.image_bytes += stored.nbytes

    def image_at(self, time: float) -> np.ndarray | None:
        self._check_open()
        image = self._images.get(time)
        if image is not None:
            self._images.move_to_end(time)
        return image

    def has_analysis(self, time: float, mode: str = "main") -> bool:
        self._check_open()
        return (time, mode) in self._analyses

    def missing_fields(self, time: float, fields, mode: str = "main") -> frozenset[str]:
        self._check_open()
        # OCR results remain reusable when only the visual analysis mode changes.
        completed = set()
        for (timestamp, _), known in self._fields.items():
            if timestamp == time:
                completed.update(known)
        return frozenset(fields) - completed

    def merge(self, observation: Observation, mode: str = "main", fields=()) -> Observation:
        self._check_open()
        time = observation.time
        if not isfinite(time):
            raise ValueError("Некорректный PTS")
        incoming = deepcopy(observation)
        current = self._observations.get(time)
        if current is None:
            current = incoming
            self._observations[time] = current
        else:
            for descriptor in dataclass_fields(Observation):
                name = descriptor.name
                previous, value = getattr(current, name), getattr(incoming, name)
                if name == "ocr_attempts":
                    for attempt in value:
                        if attempt not in previous:
                            previous.append(attempt)
                elif name in ("board", "uncertain_board"):
                    for stone in value:
                        same = next((s for s in previous if s.box == stone.box), None)
                        if same is None:
                            previous.append(stone)
                        elif same.values != stone.values and stone.stone is not None:
                            if same.stone is None:
                                previous[previous.index(same)] = stone
                            else:
                                self._conflict(time, name, same, stone)
                elif previous is None:
                    setattr(current, name, value)
                elif (
                    isinstance(previous, tuple)
                    and isinstance(value, tuple)
                    and len(previous) == len(value)
                ):
                    combined = tuple(a if a is not None else b for a, b in zip(previous, value))
                    for position, (a, b) in enumerate(zip(previous, value)):
                        if a is not None and b is not None and a != b:
                            self._conflict(time, f"{name}[{position}]", a, b)
                    setattr(current, name, combined)
                elif previous != value and value is not None:
                    if name in ("dense", "selective", "result_table"):
                        setattr(current, name, previous or value)
                    else:
                        self._conflict(time, name, previous, value)
        self._analyses.add((time, mode))
        self._fields.setdefault((time, mode), set()).update(fields)
        return current

    def _conflict(self, time, field, previous, incoming):
        conflict = dict(time=time, field=field, previous=previous, incoming=incoming)
        if conflict not in self.conflicts:
            self.conflicts.append(conflict)
