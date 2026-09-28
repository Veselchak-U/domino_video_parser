import json
import os
import tempfile
from pathlib import Path


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
                # Game exports must not overwrite an existing result.
                os.link(path, target)
        finally:
            Path(path).unlink(missing_ok=True)
