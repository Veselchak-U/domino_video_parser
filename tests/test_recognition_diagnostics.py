import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from domino_video.corrections import Corrections
from domino_video.manager import ParseManager
from domino_video.ocr_result import OCRLine, OCRResult
from domino_video.recognition_diagnostics import RecognitionDiagnostics
from domino_video.reconstruct import NameRecognitionError, ScoreRecognitionError
from domino_video.video import VideoTimeline
from domino_video.vision import Observation, ScreenRecognizer


class FixtureReader:
    def __init__(self, observations):
        self.observations = observations
        self.calls = []
        self.image = cv2.imread("tests/fixtures/frame_20.png")

    def timeline(self, path, scanning=None):
        return VideoTimeline(0, 291)

    def frames(self, path):
        for item in self.observations:
            yield item.time, item

    def frame_at(self, path, timestamp):
        self.calls.append(timestamp)
        return self.image


class FixtureRecognizer:
    def observe(self, image, timestamp, read_text):
        return image


def failing_names(observations):
    for item in observations:
        item.names = None
        item.ocr_attempts = []
    item = next(o for o in observations if 15 < o.time < 25)
    for seat, text in enumerate(["You", "", "Third", "Fourth"], 1):
        item.ocr_attempts.append(
            dict(
                field="name",
                seat=seat,
                team=None,
                time=item.time,
                raw_rows=[
                    dict(
                        text="MnX@/by" if seat == 2 else text,
                        confidence=0.725 if seat == 2 else 0.99,
                    )
                ],
                accepted_text=text,
                threshold=0.8,
                reason="low_confidence" if seat == 2 else None,
            )
        )
    return item.time


def run_fixture(tmp_path, observations, reader=None, corrections=None):
    reader = reader or FixtureReader(observations)
    manager = ParseManager(reader, FixtureRecognizer())
    source = tmp_path / "2026-09-28-13-58.mp4"
    source.write_bytes(b"fixture")
    output = tmp_path / "out"
    code = manager.run([source], output, "withoutEggs", 50, corrections or Corrections())
    report = json.loads(next((output / "report").glob("*.json")).read_text(encoding="utf-8"))
    return code, report, output, reader


def test_failed_name_has_ocr_details_and_matching_sample(tmp_path, observations, capsys):
    timestamp = failing_names(observations)
    code, report, output, reader = run_fixture(tmp_path, observations)
    assert code == 1
    (entry,) = report["games"][0]["recognition_diagnostics"]
    assert (entry["field"], entry["seat"], entry["position"]) == ("name", 2, "слева сверху")
    assert entry["time"] == timestamp
    assert entry["raw_rows"] == [dict(text="MnX@/by", confidence=0.725)]
    assert entry["accepted_text"] == "" and entry["reason"] == "low_confidence"
    assert entry["sample"]["time"] == timestamp
    sample = output / "report" / entry["sample"]["path"]
    assert np.array_equal(cv2.imread(str(sample)), reader.image[42:94, 230:410])
    assert reader.calls == [timestamp]
    assert not list(output.glob("*game*.json"))
    text = capsys.readouterr().out
    assert "слева сверху" in text and "низкая уверенность" in text
    assert str(sample) in text


def observation(time=20, attempts=None):
    return Observation(time, [], [[], [], [], []], None, False, True, ocr_attempts=attempts or [])


def read_fields(names=None, counts=None, scores=None, confidence=0.99):
    prepared = ScreenRecognizer().prepare(cv2.imread("tests/fixtures/frame_20.png"), 20, True)
    values = (
        (names or ["You", "Second", "Third", "Fourth"])
        + (counts or ["6", "6", "7"])
        + (scores or ["0/50", "0/50"])
    )
    readings = iter(
        [
            value if isinstance(value, OCRResult) else OCRResult((OCRLine(value, confidence),))
            for value in values
        ]
    )
    return prepared.finish(lambda crop: next(readings))


@pytest.mark.parametrize("confidence,accepted", [(0.799, False), (0.8, False), (0.801, True)])
def test_cpu_and_gpu_preserve_threshold_and_evidence(confidence, accepted):
    from domino_video.ocr import DirectMLAdapter

    rows = [([], "Player", confidence)]
    cpu = ScreenRecognizer()
    cpu._ocr = lambda crop: (rows, None)
    gpu = object.__new__(DirectMLAdapter)
    gpu._engine = cpu._ocr
    assert cpu._read(None) == gpu.read(None) == OCRResult((OCRLine("Player", confidence),))
    assert bool(cpu._text(None)) is accepted
    assert gpu.text(None) == cpu._text(None)


@pytest.mark.parametrize(
    "value,reason",
    [
        (OCRResult(), "no_text"),
        (OCRResult((OCRLine("bad", 0.8),)), "low_confidence"),
        ("Second", None),
    ],
)
def test_name_attempt_keeps_original_rows(value, reason):
    obs = read_fields(names=["You", value, "Third", "Fourth"])
    attempt = obs.ocr_attempts[1]
    assert attempt["reason"] == reason
    assert attempt["region"] == (230, 42, 410, 94)
    assert attempt["scale"] == 1
    assert bool(obs.names) is (reason is None)


def test_score_attempts_include_both_teams_after_first_invalid_format():
    obs = read_fields(scores=["wrong", "7/50"])
    attempts = [a for a in obs.ocr_attempts if a["field"] == "score"]
    assert [(a["team"], a["reason"]) for a in attempts] == [("A", "invalid_format"), ("B", None)]
    assert obs.scores is None
    entries = RecognitionDiagnostics().build(
        [obs], [dict(start=1, end=19)], 1, ScoreRecognitionError(1)
    )
    assert [(e["field"], e["team"], e["round"]) for e in entries] == [("score", "A", 1)]


def test_inconsistent_score_limits_are_reported_for_both_teams():
    obs = read_fields(scores=["7/50", "8/101"])
    assert obs.scores is None
    entries = RecognitionDiagnostics().build(
        [obs], [dict(start=1, end=19)], 1, ScoreRecognitionError(1)
    )
    assert [(e["team"], e["reason"]) for e in entries] == [
        ("A", "invalid_format"),
        ("B", "invalid_format"),
    ]


def test_repeated_names_report_both_positions_and_best_attempt():
    first = read_fields(names=["", "", "Third", "Fourth"])
    second = read_fields(names=["You", "Same", "Same", "Fourth"])
    second.time = 25
    for a in second.ocr_attempts:
        a["time"] = 25
    last = read_fields(names=["You", "Other", "Other", "Fourth"])
    last.time = 30
    for a in last.ocr_attempts:
        a["time"] = 30
    entries = RecognitionDiagnostics().build(
        [last, second, first], [dict(start=20, end=100)], 1, NameRecognitionError("names")
    )
    assert [(e["seat"], e["reason"], e["time"]) for e in entries] == [
        (2, "duplicate_name", 25),
        (3, "duplicate_name", 25),
    ]


@pytest.mark.parametrize(
    "error,field,count",
    [
        (NameRecognitionError("names"), "name", 4),
        (ScoreRecognitionError(1), "score", 2),
    ],
)
def test_missing_attempt_has_no_fabricated_image_or_timestamp(error, field, count):
    entries = RecognitionDiagnostics().build([observation()], [dict(start=1, end=19)], 1, error)
    assert len(entries) == count
    assert all(
        e["field"] == field
        and e["reason"] == "no_frame"
        and e["time"] is None
        and e["raw_rows"] == []
        and e["sample"] is None
        for e in entries
    )


def test_score_diagnostics_use_round_intervals_and_keep_valid_rounds():
    invalid = read_fields(scores=["bad", "bad"])
    invalid.time = 105
    for a in invalid.ocr_attempts:
        a["time"] = 105
    valid = read_fields(scores=["7/50", "8/50"])
    valid.time = 205
    group = [dict(start=20, end=100), dict(start=110, end=200)]
    entries = RecognitionDiagnostics().build([valid, invalid], group, 1, ScoreRecognitionError(1))
    assert {(e["round"], e["time"]) for e in entries} == {(1, 105)}


def test_name_interval_excludes_unrelated_attempts():
    outside = read_fields(names=["You", "Same", "Same", "Fourth"])
    outside.time = 5
    entries = RecognitionDiagnostics().build(
        [outside], [dict(start=20, end=100)], 1, NameRecognitionError("names")
    )
    assert all(e["reason"] == "no_frame" for e in entries)


def test_unreadable_count_warns_without_blocking_valid_game(tmp_path, observations):
    attempts = read_fields(counts=["oops", "6", "7"]).ocr_attempts
    next(o for o in observations if 15 < o.time < 25).ocr_attempts = attempts
    code, report, output, _ = run_fixture(tmp_path, observations)
    assert code == 0 and report["status"] == "ok"
    (entry,) = report["games"][0]["recognition_diagnostics"]
    assert (entry["field"], entry["seat"], entry["severity"]) == ("count", 2, "warning")
    assert entry["reason"] == "invalid_format" and entry["sample"]
    assert len(list(output.glob("*game*.json"))) == 1


def test_later_valid_count_eliminates_transient_failure():
    first = read_fields(counts=["oops", "6", "7"])
    second = read_fields(counts=["0", "6", "7"])
    second.time = 25
    assert RecognitionDiagnostics().build([first, second], [dict(start=20, end=100)], 1) == []


def test_corrected_names_do_not_produce_false_diagnostics(tmp_path, observations, sample_game):
    failing_names(observations)
    corrections = Corrections(
        {
            "sources": [
                dict(
                    sha256=hashlib.sha256(b"fixture").hexdigest(),
                    games=[dict(number=1, teams=sample_game["teams"], rounds=[])],
                )
            ]
        }
    )
    code, report, output, reader = run_fixture(tmp_path, observations, corrections=corrections)
    assert code == 0
    assert report["games"][0]["recognition_diagnostics"] == []
    assert not reader.calls and not (output / "report" / "samples").exists()


def test_later_full_name_set_eliminates_failure(tmp_path, observations):
    failing_names(observations)
    next(o for o in observations if 25 < o.time < 30).names = ["You", "Second", "Third", "Fourth"]
    code, report, output, reader = run_fixture(tmp_path, observations)
    assert code == 0
    assert report["games"][0]["recognition_diagnostics"] == []
    assert not reader.calls and not (output / "report" / "samples").exists()


@pytest.mark.parametrize("failure", ["frame", "encode", "publish", "changed_source"])
def test_sample_failure_keeps_original_error_and_continues(
    tmp_path, observations, monkeypatch, failure, capsys
):
    failing_names(observations)
    reader = FixtureReader(observations)
    if failure == "frame":
        monkeypatch.setattr(
            reader, "frame_at", lambda *args: (_ for _ in ()).throw(ValueError("missing frame"))
        )
    elif failure == "encode":
        monkeypatch.setattr("domino_video.storage.cv2.imencode", lambda *args: (False, None))
    elif failure == "publish":
        original = __import__("os").replace

        def replace(src, dst):
            if Path(dst).suffix == ".png":
                raise OSError("PNG denied")
            return original(src, dst)

        monkeypatch.setattr("domino_video.storage.os.replace", replace)
    else:
        original = reader.frames

        def frames(path):
            yield from original(path)
            path.write_bytes(b"changed source")

        monkeypatch.setattr(reader, "frames", frames)
    source = tmp_path / "2026-09-28-13-58.mp4"
    other = tmp_path / "2026-09-28-13-59.mp4"
    source.write_bytes(b"fixture")
    other.write_bytes(b"fixture")
    output = tmp_path / "out"
    manager = ParseManager(reader, FixtureRecognizer())
    assert manager.run([source, other], output, "withoutEggs", 50, Corrections()) == 1
    reports = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((output / "report").glob("*.json"))
    ]
    assert len(reports) == 2
    for report in reports:
        game = report["games"][0]
        assert "четыре различных имени" in game["errors"][0]["message"]
        (entry,) = game["recognition_diagnostics"]
        assert entry["sample"] is None and entry["sample_error"]
        assert entry["reason"] == "low_confidence"
    assert "образец отсутствует" in capsys.readouterr().out
    assert not list(output.rglob(".domino-*.tmp"))


def test_repeat_reuses_sample_and_json_failure_preserves_previous_references(
    tmp_path, observations, monkeypatch
):
    failing_names(observations)
    _, report, output, _ = run_fixture(tmp_path, observations)
    target = next((output / "report").glob("*.json"))
    previous = target.read_bytes()
    original_sample = (
        output / "report" / report["games"][0]["recognition_diagnostics"][0]["sample"]["path"]
    )
    sample_data = original_sample.read_bytes()
    run_fixture(tmp_path, observations)
    assert len(list((output / "report" / "samples").glob("*.png"))) == 1
    reader = FixtureReader(observations)
    reader.image[42:94, 230:410] = 0
    original = __import__("os").replace

    def replace(src, dst):
        if dst == target:
            raise OSError("report denied")
        return original(src, dst)

    monkeypatch.setattr("domino_video.storage.os.replace", replace)
    code, _, _, _ = run_fixture(tmp_path, observations, reader)
    assert code == 1
    assert target.read_bytes() == previous and original_sample.read_bytes() == sample_data
    assert len(list((output / "report" / "samples").glob("*.png"))) == 2


def test_multiple_round_errors_share_one_image_per_field(tmp_path):
    from domino_video.recognition_samples import RecognitionSamples
    from domino_video.storage import ExportStorage

    first = read_fields(scores=["bad", "7/50"])
    first.time = 105
    second = read_fields(scores=["bad", "7/50"])
    second.time = 205
    for o in (first, second):
        for a in o.ocr_attempts:
            a["time"] = o.time
    group = [dict(start=20, end=100), dict(start=110, end=200)]
    entries = RecognitionDiagnostics().build([second, first], group, 1, ScoreRecognitionError(1))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    reader = FixtureReader([])
    assert not RecognitionSamples(reader, ExportStorage()).write(
        source, hashlib.sha256(b"fixture").hexdigest(), tmp_path, entries
    )
    assert len(entries) == 2 and reader.calls == [105]
    assert entries[0]["sample"] == entries[1]["sample"]
    assert entries[1]["time"] == 205 and entries[1]["sample"]["time"] == 105
    assert len(list((tmp_path / "samples").glob("*.png"))) == 1
    actual = cv2.imread(str(tmp_path / entries[0]["sample"]["path"]))
    expected = cv2.resize(reader.image[88:135, 740:902], None, fx=3, fy=3)
    assert np.array_equal(actual, expected)


class ControlledOCRAdapter:
    def read(self, crop):
        return OCRResult((OCRLine("unreadable", 0.725),))

    def probe(self):
        pass

    def close(self):
        pass


class ControlledScreenRecognizer(ScreenRecognizer):
    def _read(self, crop):
        return ControlledOCRAdapter().read(crop)


def initialize_controlled_worker():
    from domino_video import pipeline

    pipeline._initialize_worker()
    pipeline._recognizer = ControlledScreenRecognizer()


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
@pytest.mark.parametrize("workers", [1, 2])
def test_diagnostics_and_samples_match_through_real_process_boundaries(
    tmp_path, monkeypatch, mode, workers
):
    from concurrent.futures import ProcessPoolExecutor

    from domino_video import pipeline
    from domino_video.gpu_worker import GPUWorker
    from domino_video.ocr import DeviceOCR
    from domino_video.recognition_samples import RecognitionSamples
    from domino_video.storage import ExportStorage

    def executor(**kwargs):
        kwargs["initializer"] = initialize_controlled_worker
        return ProcessPoolExecutor(**kwargs)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", executor)
    if mode == "gpu":
        monkeypatch.setattr(
            pipeline,
            "DeviceOCR",
            lambda mode, setting: DeviceOCR(
                mode,
                "1",
                available=lambda: True,
                worker_factory=lambda: GPUWorker(ControlledOCRAdapter),
            ),
        )
    image = cv2.imread("tests/fixtures/frame_20.png")

    def frames():
        for timestamp in [20, 25]:
            yield timestamp, image

    actual = list(
        pipeline.ObservationPipeline(workers, ControlledScreenRecognizer(), device=mode).observe(
            frames()
        )
    )
    expected = [ControlledScreenRecognizer().observe(image, t, True) for t in [20, 25]]
    assert actual == expected
    entries = RecognitionDiagnostics().build(
        actual, [dict(start=20, end=100)], 1, NameRecognitionError("names")
    )
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"fixture")
    reader = FixtureReader([])
    assert not RecognitionSamples(reader, ExportStorage()).write(
        source, hashlib.sha256(b"fixture").hexdigest(), tmp_path, entries
    )
    assert len(entries) == 7 and reader.calls == [20]
    from domino_video.screen_profile import ScreenProfile

    for entry in entries:
        expected_crop = ScreenProfile().crop(image, entry["field"], entry["seat"], entry["team"])
        sample = cv2.imread(str(tmp_path / entry["sample"]["path"]))
        assert np.array_equal(sample, expected_crop)


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("scale", [1, 0.75])
def test_sample_profile_matches_prepared_crops_after_rotation_and_scaling(turns, scale):
    from domino_video.screen_profile import ScreenProfile

    image = cv2.imread("tests/fixtures/frame_20.png")
    image = cv2.resize(image, None, fx=scale, fy=scale)
    image = np.rot90(image, turns)
    recognizer = ScreenRecognizer()
    prepared = recognizer.prepare(image, 20, True)
    normalized = recognizer.normalize(image)
    profile = ScreenProfile()
    for group, field, seats, teams in [
        ("names", "name", [1, 2, 3, 4], [None] * 4),
        ("counts", "count", [2, 3, 4], [None] * 3),
        ("scores", "score", [None, None], ["A", "B"]),
    ]:
        for crop, seat, team in zip(prepared.crops[group], seats, teams):
            assert np.array_equal(profile.crop(normalized, field, seat, team), crop)
