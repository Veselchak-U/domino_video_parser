"""Publish bounded diagnostic attachments after the final field checks."""

import hashlib

from .screen_profile import ScreenProfile
from .vision import ScreenRecognizer


class RecognitionSamples:
    def __init__(self, reader, storage):
        self._reader = reader
        self._storage = storage
        self._recognizer = ScreenRecognizer()
        self._profile = ScreenProfile()

    def write(self, source, expected_hash, report_dir, entries):
        chosen = {}
        for entry in entries:
            if entry["time"] is not None:
                key = entry["field"], entry["seat"], entry["team"]
                if key not in chosen or entry["time"] < chosen[key]["time"]:
                    chosen[key] = entry
        if not chosen:
            return False
        source_error = None
        try:
            with source.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != expected_hash:
                raise ValueError("Исходное видео изменилось после распознавания")
        except (OSError, ValueError) as error:
            source_error = str(error)
        images = {}
        samples = {}
        failed = False
        # Group by timestamp so only one normalized frame is held at a time.
        for key, entry in sorted(chosen.items(), key=lambda pair: pair[1]["time"]):
            try:
                if source_error:
                    raise ValueError(source_error)
                timestamp = entry["time"]
                if timestamp not in images:
                    images.clear()
                    image = self._reader.frame_at(source, timestamp)
                    images[timestamp] = self._recognizer.normalize(image)
                crop = self._profile.crop(images[timestamp], *key)
                path = self._storage.write_sample(report_dir, crop)
                samples[key] = (dict(path=path, time=timestamp), None)
            except Exception as error:
                failed = True
                samples[key] = (None, f"{type(error).__name__}: {error}")
        for entry in entries:
            key = entry["field"], entry["seat"], entry["team"]
            if key in samples and entry["time"] is not None:
                entry["sample"], sample_error = samples[key]
                if sample_error:
                    entry["sample_error"] = sample_error
        return failed

    def write_stones(self, source, expected_hash, report_dir, entries):
        chosen = [
            e
            for e in entries
            if e.get("method") in {"animation", "late_reading", "hand_difference"}
            and e.get("time") is not None
            and e.get("region")
        ]
        if not chosen:
            return False
        error = None
        try:
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                    raise ValueError("Исходное видео изменилось после распознавания")
        except (OSError, ValueError) as failure:
            error = str(failure)
        failed = False
        timestamp, image = None, None
        for entry in sorted(chosen, key=lambda e: e["time"]):
            try:
                if error:
                    raise ValueError(error)
                if timestamp != entry["time"]:
                    timestamp = entry["time"]
                    image = self._recognizer.normalize(self._reader.frame_at(source, timestamp))
                x, y, w, h = entry["region"]
                crop = image[
                    max(0, y) : min(image.shape[0], y + h), max(0, x) : min(image.shape[1], x + w)
                ]
                if not crop.size:
                    raise ValueError("Пустой участок доказательства камня")
                path = self._storage.write_sample(report_dir, crop)
                entry["sample"] = dict(path=path, time=timestamp)
            except Exception as failure:
                failed = True
                entry["sample"] = None
                entry["sample_error"] = f"{type(failure).__name__}: {failure}"
        return failed
