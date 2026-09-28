import hashlib
import time
from collections import Counter
from dataclasses import asdict

from .reconstruct import GameReconstructor, ReconstructionError
from .recording_time import RecordingTimeResolver
from .storage import ExportStorage
from .validator import GameValidator
from .video import VideoReader
from .vision import ScreenRecognizer


class ParseManager:
    def __init__(self, reader=None, recognizer=None, time_resolver=None):
        self._reader = reader or VideoReader()
        self._recognizer = recognizer or ScreenRecognizer()
        self._reconstructor = GameReconstructor()
        self._storage = ExportStorage()
        self._time_resolver = time_resolver or RecordingTimeResolver()

    def run(self, paths, output, variant, limit, corrections):
        failures = False
        hashes = {}
        hash_errors = {}
        for path in paths:
            try:
                with path.open("rb") as stream:
                    hashes[path] = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as error:
                hash_errors[path] = str(error)
        try:
            corrections.check_sources(set(hashes.values()))
        except ValueError as error:
            print(f"Ошибка исправлений: {error}")
            return 1
        for index, path in enumerate(paths, 1):
            prefix = f"undated-source-{index:03d}"
            report = dict(
                source=str(path.resolve()),
                sha256=hashes.get(path),
                status="error",
                games=[],
                errors=[],
            )
            try:
                recording_time = self._time_resolver.resolve(path)
                prefix = f"{recording_time.prefix}-source-{index:03d}"
                report["recording_time"] = asdict(recording_time)
                if path in hash_errors:
                    raise ValueError(hash_errors[path])
                print(f"[{index}/{len(paths)}] Чтение {path.name}", flush=True)
                observations = self._observe(path)
                rounds = self._reconstructor.extract(observations)
                if not rounds:
                    raise ReconstructionError(
                        "Не найдена партия поддерживаемого визуального профиля"
                    )
                groups = self._groups(rounds, observations, limit)
                corrections.check_games(hashes[path], len(groups))
                for game_number, group in enumerate(groups, 1):
                    record = dict(number=game_number, rounds=group, errors=[])
                    report["games"].append(record)
                    try:
                        changed, teams = corrections.apply(hashes[path], game_number, group)
                        names = self._names(observations, group)
                        if teams:
                            names = [
                                p["name"]
                                for p in sorted(
                                    [p for t in teams for p in t["players"]],
                                    key=lambda p: p["seat"],
                                )
                            ]
                        game = self._reconstructor.build(changed, names, variant, limit)
                        if teams:
                            # Renaming teams must also rename all score/result references.
                            game = self._rename_teams(game, teams)
                        self._check_scores(game, group, observations, limit)
                        GameValidator().validate(game)
                        target = output / f"{prefix}-game-{game_number:03d}.json"
                        self._storage.write(target, game)
                        record["output"] = str(target.resolve())
                        print(f"Сохранено: {target}", flush=True)
                    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
                        failures = True
                        record["errors"].append(dict(time=group[0]["start"], message=str(error)))
                        print(f"Партия {game_number}: {error}", flush=True)
                report["status"] = (
                    "needs_review" if any(g["errors"] for g in report["games"]) else "ok"
                )
            except Exception as error:
                # A failed decoder/OCR must not abort the remaining input files.
                failures = True
                report["errors"].append(dict(time=None, message=f"{type(error).__name__}: {error}"))
                print(f"Ошибка {path.name}: {error}", flush=True)
            report["corrections_template"] = {
                "sources": [
                    {
                        "sha256": hashes.get(path),
                        "games": [
                            {
                                "number": g["number"],
                                "rounds": [
                                    {"number": n, "events": []}
                                    for n in range(1, len(g["rounds"]) + 1)
                                ],
                            }
                            for g in report["games"]
                        ],
                    }
                ]
            }
            try:
                self._storage.write(output / f"{prefix}-report.json", report)
            except OSError as error:
                failures = True
                print(f"Не удалось сохранить отчёт: {error}", flush=True)
        return int(failures)

    def _observe(self, path):
        result = []
        last_text = -5
        last_update = time.monotonic()
        for timestamp, image in self._reader.frames(path):
            read_text = timestamp - last_text >= 5
            result.append(self._recognizer.observe(image, timestamp, read_text))
            if read_text:
                last_text = timestamp
            if time.monotonic() - last_update >= 20:
                print(f"  Обработано {timestamp:.0f} с видео", flush=True)
                last_update = time.monotonic()
        return result

    def _groups(self, rounds, observations, limit):
        groups = []
        current = []
        for i, rnd in enumerate(rounds):
            current.append(rnd)
            stop = rounds[i + 1]["start"] if i + 1 < len(rounds) else float("inf")
            scores = [
                o.scores
                for o in observations
                if rnd["end"] is not None and rnd["end"] < o.time < stop and o.scores
            ]
            if any(max(score) >= limit for score in scores):
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    def _names(self, observations, group):
        start = group[0]["start"] - 10
        end = group[-1]["end"] or float("inf")
        names = Counter(tuple(o.names) for o in observations if start <= o.time <= end and o.names)
        return list(names.most_common(1)[0][0]) if names else None

    def _check_scores(self, game, group, observations, limit):
        for i, (rnd, raw) in enumerate(zip(game["rounds"], group)):
            stop = group[i + 1]["start"] if i + 1 < len(group) else raw["end"] + 16
            observed = [o for o in observations if raw["end"] < o.time < stop and o.scores]
            if not observed:
                raise ReconstructionError(f"Кон {i + 1}: не прочитан итоговый счёт")
            # The new score appears after the reveal animation. Prefer a final score
            # over a following reset to 0:0 when another match starts immediately.
            terminal = [o for o in observed if max(o.scores) >= limit]
            last = (terminal or observed)[-1]
            if last.limit != limit:
                raise ReconstructionError("Лимит табло расходится с параметром запуска")
            team_names = [t["name"] for t in game["teams"]]
            expected = tuple(rnd["result"]["total_score"][t] for t in team_names)
            if expected != last.scores:
                raise ReconstructionError(
                    f"Кон {i + 1}: на табло {last.scores}, по правилам {expected}"
                )

    def _rename_teams(self, game, teams):
        mapping = {game["teams"][i]["name"]: teams[i]["name"] for i in range(2)}
        game["teams"] = teams
        for rnd in game["rounds"]:
            result = rnd["result"]
            for field in ("round_score", "total_score", "fence"):
                if field in result:
                    result[field] = {mapping[k]: v for k, v in result[field].items()}
            if result["winner_team"] is not None:
                result["winner_team"] = mapping[result["winner_team"]]
        if "winner" in game["result"]:
            game["result"]["winner"] = mapping[game["result"]["winner"]]
        return game
