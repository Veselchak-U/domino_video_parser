import hashlib
import json
import os
import shutil
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
    def move_video(self, source: Path, ready_dir: Path):
        if source.resolve().parent == ready_dir.resolve():
            return source
        ready_dir.mkdir(parents=True, exist_ok=True)
        target = ready_dir / source.name
        number = 0
        while True:
            try:
                destination = target.open("xb")
                break
            except (FileExistsError, IsADirectoryError, PermissionError):
                if not os.path.lexists(target):
                    raise
                number += 1
                target = ready_dir / f"{source.stem}-{number:03d}{source.suffix}"
        try:
            with destination:
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            source.unlink()
        except (Exception, KeyboardInterrupt):
            target.unlink(missing_ok=True)
            raise
        return target

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
