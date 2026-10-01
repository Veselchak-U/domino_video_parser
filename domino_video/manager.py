import hashlib
from contextlib import closing
from dataclasses import asdict

from .pipeline import ObservationPipeline
from .player_names import PlayerNameResolver
from .progress import ConsoleProgress
from .recognition_diagnostics import RecognitionDiagnostics
from .recognition_samples import RecognitionSamples
from .reconstruct import GameReconstructor, ReconstructionError, ScoreRecognitionError
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
        self._diagnostics = RecognitionDiagnostics()
        self._samples = RecognitionSamples(self._reader, self._storage)
        self._time_resolver = time_resolver or RecordingTimeResolver()

    def run(
        self,
        paths,
        output,
        variant,
        limit,
        corrections,
        workers=1,
        device="cpu",
        gpu_workers="auto",
    ):
        progress = ConsoleProgress()
        progress.message(f"Процессов распознавания: {workers}")
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
            saved_paths = []
            with progress:
                progress.start(index, len(paths), path.name)
                try:
                    recording_time = self._time_resolver.resolve(path)
                    prefix = f"{recording_time.prefix}-source-{index:03d}"
                    report["recording_time"] = asdict(recording_time)
                    if path in hash_errors:
                        raise ValueError(hash_errors[path])
                    observations = self._observe(path, workers, progress, device, gpu_workers)
                    progress.update(99, "проверка и сохранение")
                    rounds = self._reconstructor.extract(observations)
                    windows = self._reconstructor.recovery_windows(rounds)
                    if windows and hasattr(self._reader, "interval_frames"):
                        progress.message(f"Дополнительное чтение камней: {len(windows)} интервалов")
                        extra = []
                        with closing(self._reader.interval_frames(path, windows)) as frames:
                            for timestamp, image in frames:
                                observed = self._recognizer.prepare(
                                    image, timestamp, False, read_motion=True
                                ).observation
                                observed.dense = True
                                extra.append(observed)
                        self._reconstructor.integrate_recovery(rounds, extra)
                    if not rounds:
                        raise ReconstructionError(
                            "Не найдена партия поддерживаемого визуального профиля"
                        )
                    groups = self._groups(rounds, observations, limit)
                    corrections.check_games(hashes[path], len(groups))
                    for game_number, group in enumerate(groups, 1):
                        record = dict(number=game_number, rounds=group, errors=[])
                        report["games"].append(record)
                        game_error = None
                        resolved = None
                        diagnostic_group = group
                        try:
                            changed, teams = corrections.apply(hashes[path], game_number, group)
                            diagnostic_group = changed
                            resolved = PlayerNameResolver().resolve(observations, group)
                            names = resolved.names
                            if teams:
                                resolved = None
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
                            for rnd in changed:
                                for entry in rnd.get("stone_recovery", []):
                                    entry["status"] = "validated"
                            target = output / f"{prefix}-game-{game_number:03d}.json"
                            self._storage.write(target, game, replace=True)
                            record["output"] = str(target.resolve())
                            saved_paths.append(target)
                        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
                            game_error = error
                            failures = True
                            record["errors"].append(
                                dict(time=group[0]["start"], message=str(error))
                            )
                            progress.message(f"Партия {game_number}: {error}")
                        entries = self._diagnostics.build(
                            observations, group, game_number, game_error, resolved
                        )
                        record["recognition_diagnostics"] = entries
                        stone_entries = []
                        for round_number, rnd in enumerate(diagnostic_group, 1):
                            for entry in rnd.get("stone_recovery", []):
                                entry.update(game=game_number, round=round_number)
                                stone_entries.append(entry)
                            for event in rnd.get("unresolved", []):
                                if event.get("stone") is None:
                                    stone_entries.append(
                                        dict(
                                            event,
                                            game=game_number,
                                            round=round_number,
                                            method=None,
                                            sample=None,
                                        )
                                    )
                        record["stone_recovery"] = stone_entries
                        stone_sample_failed = self._samples.write_stones(
                            path, hashes[path], output / "report", stone_entries
                        )
                        sample_failed = self._samples.write(
                            path, hashes[path], output / "report", entries
                        )
                        if sample_failed or stone_sample_failed:
                            failures = True
                            record["errors"].append(
                                dict(
                                    time=None,
                                    message="Не удалось приложить образец OCR; подробности в recognition_diagnostics",
                                )
                            )
                        for entry in entries:
                            progress.message(self._diagnostics.message(entry, output / "report"))
                    report["status"] = (
                        "needs_review" if any(g["errors"] for g in report["games"]) else "ok"
                    )
                except KeyboardInterrupt:
                    progress.message("Обработка прервана")
                    progress.finish(False)
                    return 1
                except Exception as error:
                    # A failed decoder/OCR must not abort the remaining input files.
                    failures = True
                    report["errors"].append(
                        dict(time=None, message=f"{type(error).__name__}: {error}")
                    )
                    progress.message(f"Ошибка {path.name}: {error}")
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
                report_saved = False
                try:
                    self._storage.write(
                        output / "report" / f"{prefix}-report.json", report, replace=True
                    )
                    report_saved = True
                except KeyboardInterrupt:
                    progress.message("Обработка прервана")
                    return 1
                except OSError as error:
                    failures = True
                    progress.message(f"Не удалось сохранить отчёт: {error}")
                success = report_saved and report["status"] == "ok"
                if success:
                    progress.update(99, "перенос видео")
                    try:
                        ready_dir = (
                            path.parent
                            if path.parent.name.casefold() == "ready"
                            else path.parent / "ready"
                        )
                        ready_path = self._storage.move_video(path, ready_dir)
                        progress.message(f"Видео готово: {ready_path}")
                    except KeyboardInterrupt:
                        progress.message("Обработка прервана")
                        progress.finish(False)
                        return 1
                    except OSError as error:
                        failures = True
                        success = False
                        progress.message(f"Не удалось переместить видео {path.name}: {error}")
                progress.finish(success)
                for target in saved_paths:
                    progress.message(f"Сохранено: {target}")
        return int(failures)

    def _observe(self, path, workers, progress, device="cpu", gpu_workers="auto"):
        timeline = self._reader.timeline(
            path, lambda: progress.update(0, "определение длительности")
        )

        def ready(message):
            progress.message(message)
            progress.ready()

        pipeline = ObservationPipeline(
            workers,
            recognizer=self._recognizer,
            device=device,
            message=ready,
            gpu_workers=gpu_workers,
        )
        result = []
        with closing(pipeline.observe(self._reader.frames(path))) as observations:
            for observation in observations:
                result.append(observation)
                processed = max(0, min(timeline.duration, observation.time - timeline.start))
                progress.update(processed / timeline.duration * 100, processed_seconds=processed)
        progress.update(99, processed_seconds=timeline.duration)
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

    def _check_scores(self, game, group, observations, limit):
        for i, (rnd, raw) in enumerate(zip(game["rounds"], group)):
            stop = group[i + 1]["start"] if i + 1 < len(group) else raw["end"] + 16
            observed = [o for o in observations if raw["end"] < o.time < stop and o.scores]
            if not observed:
                raise ScoreRecognitionError(i + 1)
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
