import argparse
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
    ".mts",
    ".m2ts",
}


class RussianParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"Ошибка параметров (проверьте обязательные аргументы): {message}\n")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = RussianParser(description="Распознать записи экранной игры домино локально.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("in"),
        help="каталог видео (по умолчанию: in; без вложенных папок)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("out"), help="каталог результатов (по умолчанию: out)"
    )
    parser.add_argument(
        "--fish-variant", required=True, choices=["withEggs", "withoutEggs"], help="правила рыбы"
    )
    parser.add_argument(
        "--score-limit", required=True, type=int, choices=[50, 101], help="лимит партии"
    )
    parser.add_argument("--corrections", type=Path, help="JSON исправлений из отчёта")
    from .pipeline import WorkerSettings

    settings = WorkerSettings()
    parser.add_argument(
        "--workers",
        type=settings.parse,
        default=settings.default(),
        help="число процессов распознавания (по умолчанию: доступные логические CPU; Windows: максимум 61)",
    )
    args = parser.parse_args(argv)
    try:
        videos = sorted(
            (
                path
                for path in args.input.iterdir()
                if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file()
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except OSError as error:
        print(f"Ошибка чтения входного каталога {args.input}: {error}", file=sys.stderr)
        return 1
    if not videos:
        print(
            f"Ошибка: в каталоге {args.input} нет видеофайлов поддерживаемых расширений.",
            file=sys.stderr,
        )
        return 1
    from .corrections import Corrections
    from .manager import ParseManager
    from .storage import read_json

    try:
        corrections = (
            Corrections(read_json(args.corrections.read_text(encoding="utf-8")))
            if args.corrections
            else Corrections()
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Не удалось прочитать исправления: {error}")
    return ParseManager().run(
        videos, args.output, args.fish_variant, args.score_limit, corrections, args.workers
    )
