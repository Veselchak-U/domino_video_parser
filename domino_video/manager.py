import hashlib
import heapq
from contextlib import closing
from dataclasses import asdict, is_dataclass, replace

from .evidence import SourceEvidence
from .partial_game import build_partial_game, validate_partial_game
from .pipeline import ObservationPipeline
from .player_names import PlayerNameResolver
from .progress import ConsoleProgress
from .recognition_diagnostics import RecognitionDiagnostics
from .recognition_plan import (
    OCRFields,
    recognition_requests,
    score_transition_windows,
    score_visible,
)
from .recognition_samples import RecognitionSamples
from .reconstruct import GameReconstructor, ReconstructionError, ScoreRecognitionError
from .recording_time import RecordingTimeResolver
from .remaining_evidence import confirm_empty_remaining
from .round_start import confirm_round_starts, start_requests, start_windows
from .storage import ExportStorage
from .validator import GameValidator
from .video import VideoReader
from .vision import ScreenRecognizer


class ParseManager:
    def __init__(self, reader=None, recognizer=None, time_resolver=None):
        self._custom_reader = reader is not None
        self._reader = reader or VideoReader()
        self._recognizer = recognizer or ScreenRecognizer()
        self._reconstructor = GameReconstructor()
        self._storage = ExportStorage()
        self._diagnostics = RecognitionDiagnostics()
        self._samples = RecognitionSamples(self._reader, self._storage)
        self._time_resolver = time_resolver or RecordingTimeResolver()
        self._evidence = None

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
        if not self._custom_reader:
            self._reader = VideoReader(workers, device, progress.message)
            self._samples = RecognitionSamples(self._reader, self._storage)
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
            if hasattr(corrections, "check_settings"):
                corrections.check_settings(variant, limit)
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
            with progress, SourceEvidence(hashes.get(path, "")) as evidence:
                self._evidence = evidence
                self._score_searched = False
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
                    if any(o.selective for o in observations):
                        self._read_start_frames(path, rounds, observations, workers)
                    for group in self._groups(rounds, observations, limit):
                        self._reconstructor.resolve_logically(group, observations, variant, limit)
                    if any(o.selective for o in observations):
                        self._read_fields(
                            path, rounds, observations, workers, progress, device, gpu_workers
                        )
                    for group in self._groups(rounds, observations, limit):
                        self._reconstructor.resolve_logically(group, observations, variant, limit)
                    windows = self._reconstructor.recovery_windows(rounds)
                    if windows and (
                        hasattr(self._reader, "sampled_interval_frames")
                        or hasattr(self._reader, "interval_frames")
                    ):
                        progress.message(
                            f"Дополнительное чтение камней: 10 кадров/сек, {len(windows)} интервалов"
                        )
                        pipeline = ObservationPipeline(
                            workers, recognizer=self._recognizer, motion=True
                        )
                        with closing(
                            pipeline.observe(self._recovery_frames(path, windows))
                        ) as rows:
                            extra = sorted(
                                (evidence.merge(obs, "motion") for obs in rows),
                                key=lambda o: o.time,
                            )
                        by_time = {o.time: o for o in observations}
                        by_time.update({o.time: o for o in extra})
                        observations[:] = sorted(by_time.values(), key=lambda o: o.time)
                        self._reconstructor.integrate_recovery(rounds, extra)
                        for rnd in rounds:
                            if not rnd.get("logically_resolved"):
                                rnd.setdefault("logical_resolution", []).append(
                                    dict(
                                        stage="local-10fps",
                                        intervals=[
                                            [start, stop]
                                            for start, stop in windows
                                            if start <= (rnd.get("end") or float("inf"))
                                            and stop >= rnd["start"]
                                        ],
                                        reason="Логическая проверка не дала единственный журнал",
                                    )
                                )
                    if any(o.selective for o in observations):
                        self._read_fields(
                            path, rounds, observations, workers, progress, device, gpu_workers
                        )
                    if not rounds:
                        raise ReconstructionError(
                            "Не найдена партия поддерживаемого визуального профиля",
                            code="no_supported_game",
                        )
                    groups = self._groups(rounds, observations, limit)
                    for group in groups:
                        self._reconstructor.resolve_logically(group, observations, variant, limit)
                    if any(o.selective for o in observations):
                        for n, group in enumerate(groups):
                            group[-1]["names_end"] = (
                                groups[n + 1][0]["start"]
                                if n + 1 < len(groups)
                                else max(o.time for o in observations) + 0.001
                            )
                    corrections.check_games(hashes[path], len(groups))
                    for game_number, group in enumerate(groups, 1):
                        record = dict(
                            number=game_number,
                            rounds=group,
                            errors=[],
                            reconstruction_diagnostics=[],
                        )
                        report["games"].append(record)
                        game_error = None
                        resolved = None
                        names = None
                        detected_names = None
                        diagnostic_group = group
                        try:
                            changed, teams = corrections.apply(hashes[path], game_number, group)
                            diagnostic_group = changed
                            resolved = PlayerNameResolver().resolve(observations, group)
                            names = resolved.names
                            detected_names = names
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
                            conflicts = self._evidence_conflicts()
                            record["evidence_conflicts"] = conflicts
                            material = [
                                item
                                for item in conflicts
                                if item["field"] == "board"
                                or item["field"].startswith(("scores", "counts"))
                                if group[0]["start"]
                                <= item["time"]
                                <= (group[-1]["end"] or float("inf")) + 16
                            ]
                            if material:
                                raise ReconstructionError(
                                    "Противоречивые чтения одного исходного кадра",
                                    code="observation_conflict",
                                    details=dict(conflicts=material),
                                )
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
                            failure = dict(time=None, message=str(error))
                            if isinstance(error, ReconstructionError):
                                diagnostic = dict(error.diagnostic, game=game_number)
                                record["reconstruction_diagnostics"].append(diagnostic)
                                failure.update(
                                    {
                                        key: diagnostic[key]
                                        for key in ("code", "round", "event", "time")
                                    }
                                )
                            record["errors"].append(failure)
                            progress.message(f"Партия {game_number}: {error}")
                        if game_error is not None:
                            try:
                                partial = build_partial_game(
                                    diagnostic_group,
                                    names if names and len(set(names)) == 4 else detected_names,
                                    variant,
                                    limit,
                                    hashes[path],
                                    game_number,
                                    record["reconstruction_diagnostics"],
                                )
                                validate_partial_game(partial)
                                partial_target = (
                                    output / f"{prefix}-game-{game_number:03d}-needs-review.json"
                                )
                                self._storage.write(partial_target, partial, replace=True)
                                record["partial_output"] = str(partial_target.resolve())
                                for raw, partial_round in zip(diagnostic_group, partial["rounds"]):
                                    unknown_ids = [
                                        move["event_id"]
                                        for move in partial_round["moves"]
                                        if move["recognition_status"] == "unresolved"
                                    ]
                                    if unknown_ids:
                                        raw.setdefault("logical_resolution", []).append(
                                            dict(
                                                stage="unresolved",
                                                events=unknown_ids,
                                                reason=str(game_error),
                                            )
                                        )
                                for rnd in partial["rounds"]:
                                    for move in rnd["moves"]:
                                        if move["recognition_status"] == "unresolved":
                                            progress.message(
                                                f"Нераспознанный ход: партия {game_number}, "
                                                f"кон {rnd['number']}, {move['event_id']}, "
                                                f"интервал {move['interval']}: {move['reason']}; "
                                                f"JSON: {partial_target}"
                                            )
                            except (ValueError, OSError, KeyError, TypeError) as partial_error:
                                record["errors"].append(
                                    dict(time=None, message=f"Неполный JSON: {partial_error}")
                                )
                                progress.message(
                                    f"Не удалось сохранить неполный JSON: {partial_error}"
                                )
                        record["logical_resolution"] = [
                            dict(entry, round=number)
                            for number, rnd in enumerate(diagnostic_group, 1)
                            for entry in rnd.get("logical_resolution", [])
                        ]
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
                    if isinstance(error, ReconstructionError):
                        report.setdefault("reconstruction_diagnostics", []).append(error.diagnostic)
                        report["errors"][-1].update(
                            code=error.diagnostic["code"], time=error.diagnostic["time"]
                        )
                    progress.message(f"Ошибка {path.name}: {error}")
                report["evidence_conflicts"] = self._evidence_conflicts()
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

    def _evidence_conflicts(self):
        if self._evidence is None:
            return []

        def serializable(value):
            if is_dataclass(value):
                return asdict(value)
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [serializable(item) for item in value]
            return value

        return [serializable(item) for item in self._evidence.conflicts]

    def _cached_frames(self, frames):
        with closing(frames):
            for timestamp, image in frames:
                if self._evidence is not None and hasattr(image, "nbytes"):
                    self._evidence.cache_image(timestamp, image)
                yield timestamp, image

    def _selected_frames(self, path, timestamps):
        times = sorted(set(timestamps))
        cached = {
            timestamp: self._evidence.image_at(timestamp)
            for timestamp in times
            if self._evidence is not None
        }
        missing = [timestamp for timestamp in times if cached.get(timestamp) is None]

        def decoded():
            if hasattr(self._reader, "selected_frames"):
                yield from self._reader.selected_frames(path, missing)
            else:
                for timestamp in missing:
                    yield timestamp, self._reader.frame_at(path, timestamp)

        with closing(decoded()) as source:
            for timestamp in times:
                if cached.get(timestamp) is not None:
                    yield timestamp, cached[timestamp]
                else:
                    actual, image = next(source)
                    if actual != timestamp:
                        raise ValueError("Адресное чтение изменило исходный PTS")
                    yield actual, image

    def _recovery_frames(self, path, windows):
        existing = [
            obs.time
            for obs in self._evidence.observations
            if any(start <= obs.time <= stop for start, stop in windows)
            and not self._evidence.has_analysis(obs.time, "motion")
            and (
                hasattr(self._reader, "selected_frames")
                or hasattr(self._reader, "frame_at")
                or self._evidence.image_at(obs.time) is not None
            )
        ]

        def sampled():
            if hasattr(self._reader, "sampled_interval_frames"):
                yield from self._reader.sampled_interval_frames(path, windows, sample_fps=10)
            else:
                # Custom readers can retain the native API; sample the iterable
                # here rather than silently returning to a native motion pass.
                last = float("-inf")
                with closing(self._reader.interval_frames(path, windows)) as frames:
                    for timestamp, image in frames:
                        if timestamp + 1e-9 >= last + 0.1:
                            last = timestamp
                            yield timestamp, image

        selected = self._selected_frames(path, existing)
        # Every yielded result is ordered by source time, including coarse
        # observations outside the new regular grid. Images stay bounded.
        seen = set()
        with closing(sampled()) as dense, closing(selected) as ordinary:
            merged = heapq.merge(dense, ordinary, key=lambda item: item[0])
            for timestamp, image in merged:
                if timestamp in seen or self._evidence.has_analysis(timestamp, "motion"):
                    continue
                seen.add(timestamp)
                cached = self._evidence.image_at(timestamp)
                if cached is not None:
                    image = cached
                if hasattr(image, "nbytes"):
                    self._evidence.cache_image(timestamp, image)
                yield timestamp, image

    def _observe(self, path, workers, progress, device="cpu", gpu_workers="auto"):
        timeline = self._reader.timeline(
            path, lambda: progress.update(0, "определение длительности")
        )

        def ready(message):
            progress.message("Анализ камней: 2 кадра/сек")
            progress.ready()

        pipeline = ObservationPipeline(
            workers,
            recognizer=self._recognizer,
            device="cpu",
            message=ready,
            gpu_workers=gpu_workers,
            fields={},
        )
        result = []
        with closing(
            pipeline.observe(self._cached_frames(self._reader.frames(path)))
        ) as observations:
            for observation in observations:
                if self._evidence is not None:
                    known = [
                        name
                        for name in ("names", "counts", "scores", "reveal_points")
                        if getattr(observation, name) is not None
                    ]
                    observation = self._evidence.merge(observation, fields=known)
                result.append(observation)
                processed = max(0, min(timeline.duration, observation.time - timeline.start))
                progress.update(processed / timeline.duration * 100, processed_seconds=processed)
        progress.update(99, processed_seconds=timeline.duration)
        return result

    def _read_fields(self, path, rounds, observations, workers, progress, device, gpu_workers):
        requests = recognition_requests(rounds, observations)
        for timestamp in start_requests(rounds, observations):
            requests[timestamp] = replace(requests.get(timestamp, OCRFields()), counts=True)
        score_times = []
        if not getattr(self, "_score_searched", False):
            score_times = self._score_transition_frames(
                path, rounds, observations, workers, progress
            )
            self._score_searched = True
        # Keep each transition's candidates even if a later empty-looking UI
        # screen would displace them from the ordinary last-three budget.
        for timestamp in score_times:
            requests[timestamp] = replace(requests.get(timestamp, OCRFields()), scores=True)
        if self._evidence is not None:
            trimmed = {}
            for timestamp, fields in requests.items():
                desired = [name for name, enabled in asdict(fields).items() if enabled]
                missing = self._evidence.missing_fields(timestamp, desired)
                if missing:
                    trimmed[timestamp] = OCRFields(**dict.fromkeys(missing, True))
            requests = trimmed
        if not requests:
            return
        progress.message(f"Адресное чтение текста: {len(requests)} кадров")

        def frames():
            yield from self._cached_frames(self._selected_frames(path, requests))

        pipeline = ObservationPipeline(
            workers,
            recognizer=self._recognizer,
            device=device,
            message=progress.message,
            gpu_workers=gpu_workers,
            fields=requests,
        )
        by_time = {o.time: o for o in observations}
        with closing(pipeline.observe(frames())) as rows:
            for obs in rows:
                original = by_time[obs.time]
                if self._evidence is not None:
                    by_time[obs.time] = self._evidence.merge(
                        obs,
                        fields=[
                            name for name, value in asdict(requests[obs.time]).items() if value
                        ],
                    )
                else:
                    for attempt in obs.ocr_attempts:
                        if attempt not in original.ocr_attempts:
                            original.ocr_attempts.append(attempt)
                    for name in (
                        "names",
                        "scores",
                        "limit",
                        "counts",
                        "reveal_points",
                        "reveal_valid",
                    ):
                        value = getattr(obs, name)
                        if value is not None:
                            setattr(original, name, value)
        observations[:] = sorted(by_time.values(), key=lambda o: o.time)
        for rnd in rounds:
            rnd["counters"] = [
                dict(time=o.time, counts=o.counts)
                for o in observations
                if rnd["start"] <= o.time < (rnd["end"] or float("inf")) and o.counts is not None
            ]
        confirm_empty_remaining(rounds, observations)
        confirm_round_starts(rounds, observations)

    def _read_start_frames(self, path, rounds, observations, workers):
        if not rounds or not hasattr(self._reader, "interval_frames"):
            return

        def frames():
            last = float("-inf")
            with closing(self._reader.interval_frames(path, start_windows(rounds))) as source:
                for timestamp, image in source:
                    if timestamp - last >= 0.09:
                        last = timestamp
                        yield timestamp, image

        pipeline = ObservationPipeline(workers, recognizer=self._recognizer, motion=True)
        by_time = {o.time: o for o in observations}
        with closing(pipeline.observe(self._cached_frames(frames()))) as rows:
            for obs in rows:
                if not obs.board and len(set(obs.hands[0])) == 7:
                    if self._evidence is not None:
                        obs = self._evidence.merge(obs, "motion")
                    by_time.setdefault(obs.time, obs)
        observations[:] = sorted(by_time.values(), key=lambda o: o.time)

    def _score_transition_frames(self, path, rounds, observations, workers, progress):
        windows = score_transition_windows(rounds, observations)
        if not windows or not hasattr(self._reader, "interval_frames"):
            return []
        progress.message(f"Поиск итогового табло: {len(windows)} интервалов")
        pipeline = ObservationPipeline(
            workers, recognizer=self._recognizer, fields={}, device="cpu"
        )
        candidates = [[] for _ in windows]
        with closing(pipeline.observe(self._reader.interval_frames(path, windows))) as rows:
            for obs in rows:
                if not score_visible(obs):
                    continue
                for interval, selected in zip(windows, candidates):
                    if interval[0] < obs.time < interval[1]:
                        selected.append(obs)
                        del selected[:-3]
        by_time = {o.time: o for o in observations}
        selected_times = set()
        for selected in candidates:
            for obs in selected:
                selected_times.add(obs.time)
                if obs.time not in by_time:
                    obs.dense = True
                    if self._evidence is not None:
                        obs = self._evidence.merge(obs)
                    by_time[obs.time] = obs
        observations[:] = sorted(by_time.values(), key=lambda o: o.time)
        return sorted(selected_times)

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
                raise ReconstructionError(
                    "Лимит табло расходится с параметром запуска",
                    code="score_limit_conflict",
                    round_number=i + 1,
                    time=last.time,
                    details=dict(expected=limit, observed=last.limit),
                )
            team_names = [t["name"] for t in game["teams"]]
            expected = tuple(rnd["result"]["total_score"][t] for t in team_names)
            if expected != last.scores:
                raise ReconstructionError(
                    f"Кон {i + 1}: на табло {last.scores}, по правилам {expected}",
                    code="score_conflict",
                    round_number=i + 1,
                    time=last.time,
                    details=dict(expected=expected, observed=last.scores),
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
