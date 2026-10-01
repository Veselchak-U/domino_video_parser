import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Повтор ключа JSON: {key}")
        result[key] = value
    return result


def read_json(text):
    return json.loads(text, object_pairs_hook=_unique_object)


class ExportStorage:
    def write_sample(self, report_dir: Path, image):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Не удалось закодировать образец PNG")
        data = encoded.tobytes()
        relative = Path("samples") / f"{hashlib.sha256(data).hexdigest()}.png"
        target = report_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() == data:
            return relative.as_posix()
        fd, path = tempfile.mkstemp(prefix=".domino-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path, target)
        finally:
            Path(path).unlink(missing_ok=True)
        return relative.as_posix()

    def write(self, target: Path, document, *, replace=False):
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        fd, path = tempfile.mkstemp(prefix=".domino-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(path, target)
            else:
                # Callers must explicitly opt into replacing an existing file.
                os.link(path, target)
        finally:
            Path(path).unlink(missing_ok=True)
