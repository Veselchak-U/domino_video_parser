"""Integration acceptance check against an explicitly supplied video, offline."""

import argparse
import hashlib
import json
import shutil
import socket
import tempfile
from pathlib import Path


def blocked(*args, **kwargs):
    raise AssertionError("Обработка попыталась открыть сетевое соединение")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    args = parser.parse_args()
    with args.video.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    from domino_video.cli import main as parse

    # Isolate the sample so other videos beside it are not processed.
    with tempfile.TemporaryDirectory(prefix="domino-acceptance-") as directory:
        shutil.copyfile(args.video, Path(directory) / args.video.name)
        code = parse(
            [
                "--input",
                directory,
                "--output",
                str(args.output),
                "--fish-variant",
                "withoutEggs",
                "--score-limit",
                "50",
            ]
        )
    if code:
        raise SystemExit(code)
    results = list(args.output.glob("*-game-*.json"))
    assert len(results) == 1
    actual = json.loads(results[0].read_text(encoding="utf-8"))
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    assert actual == expected, "Журнал отличается от ручного эталона"
    with args.video.open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    assert before == after, "Исходное видео изменено"
    print(f"Подтверждены все раздачи/ходы/итоги, обработка без сети, SHA-256 видео {after}")


if __name__ == "__main__":
    main()
