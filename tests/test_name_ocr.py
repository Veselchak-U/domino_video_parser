import numpy as np
import pytest

from domino_video.name_ocr import CTCDecoder, NameOCR, NameReading, NameSymbol


def predictions(ids, probabilities, classes=4):
    data = np.zeros((1, len(ids), classes), dtype=np.float32)
    for i, (token, confidence) in enumerate(zip(ids, probabilities)):
        data[0, i, token] = confidence
    return data


def test_ctc_uses_real_token_confidence_and_blank_separates_repeats():
    symbols = CTCDecoder().decode(
        predictions([0, 1, 1, 0, 1, 2, 0], [1, 0.81, 0.99, 1, 0.69, 0.9, 1]),
        ["blank", "A", "Б", " "],
        70,
    )
    assert [s.text for s in symbols] == ["A", "A", "Б"]
    assert [s.confidence for s in symbols] == pytest.approx([0.81, 0.69, 0.9])
    assert all(s.end > s.start for s in symbols)
    assert NameReading(symbols=symbols).text == "A*Б"


def test_blank_result_does_not_invent_characters():
    assert CTCDecoder().decode(predictions([0, 0], [1, 1]), ["blank", "A", "Б", " "], 20) == ()


def test_supplementary_punctuation_cannot_overwrite_confident_cyrillic():
    reader = object.__new__(NameOCR)
    primary = (NameSymbol("М", 0.9, 0, 10), NameSymbol("?", 0.4, 10, 20))
    other = (
        NameSymbol("M", 0.99, 0, 10),
        NameSymbol("@", 0.99, 10, 20),
        NameSymbol("!", 0.9, 20, 30),
    )
    merged = reader.merge(primary, other)
    assert NameReading(symbols=merged).text == "М@!"


@pytest.mark.parametrize("end,expected", [(20, "@"), (20.01, "*")])
def test_symbol_alignment_iou_boundary(end, expected):
    reader = object.__new__(NameOCR)
    primary = (NameSymbol("?", 0.2, 0, 10),)
    other = (NameSymbol("@", 0.99, 0, end),)
    assert NameReading(symbols=reader.merge(primary, other)).text == expected


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_model_resource_failure_is_explicit(tmp_path, fault):
    p = tmp_path / "model.onnx"
    if fault == "corrupt":
        p.write_bytes(b"not a model")
    with pytest.raises(RuntimeError, match="модел"):
        NameOCR.model_path(p)


def test_packaged_model_has_full_alphabet_and_runs_offline(monkeypatch):
    import socket

    from rapidocr_onnxruntime import RapidOCR

    def blocked(*args, **kwargs):
        raise AssertionError("Network access during OCR initialization")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    reader = NameOCR(RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1))
    assert set("ЁёABCxyz0123456789").issubset(reader.characters)
    result = reader._predict(
        np.ones((48, 180, 3), np.uint8) * 255, reader.session, reader.characters, 48, 320
    )
    assert isinstance(result, tuple)
    reader.close()
    assert reader.session is reader.engine is None


def test_low_confidence_lines_are_not_removed_before_symbol_decoding():
    reader = object.__new__(NameOCR)
    calls = []

    class Session:
        pass

    class Secondary:
        session = Session()
        session.session = object()
        postprocess_op = Session()
        postprocess_op.character = ["blank", "A", " "]
        rec_image_shape = [3, 48, 320]

    class Engine:
        text_rec = Secondary()

        def __call__(self, crop, **kwargs):
            calls.append(kwargs)
            return [[[0, 0], [20, 0], [20, 10], [0, 10]]], None

        def get_crop_img_list(self, crop, boxes):
            return [crop]

    reader.engine, reader.session, reader.characters = Engine(), object(), []
    reader._predict = lambda *args: (NameSymbol("A", 0.1, 0, 10), NameSymbol("!", 0.9, 10, 20))
    reading = reader.read(np.zeros((10, 20, 3), np.uint8))
    assert reading.text == "*!"
    assert calls == [dict(use_rec=False, use_cls=False)]
    assert all(row.confidence < 0.8 for row in reading.rows)


@pytest.mark.parametrize("text", ["علي", "राज", "דוד", "e\u0301"])
def test_confident_foreign_logical_order_is_not_reversed(text):
    other = tuple(NameSymbol(c, 0.99, len(text) - i, len(text) - i + 1) for i, c in enumerate(text))
    reader = object.__new__(NameOCR)
    assert reader.merge((), other) == other


@pytest.mark.parametrize("failed_stage", [None, 1, 2, 3, 4])
def test_directml_probes_and_closes_every_session_on_stage_failure(monkeypatch, failed_stage):
    from types import SimpleNamespace

    from rapidocr_onnxruntime import RapidOCR

    from domino_video.ocr import DirectMLAdapter

    engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    monkeypatch.setattr("rapidocr_onnxruntime.RapidOCR", lambda **kwargs: engine)
    sessions = []
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" + (
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    )

    class Session:
        def __init__(self, path, **kwargs):
            self.number = len(sessions) + 1
            self.runs, self.closed, self.fallback_disabled = [], False, False
            sessions.append(self)

        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def get_outputs(self):
            return [SimpleNamespace(shape=[1, 40, len(chars) + 2])]

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={"character": "\n".join(chars)})

        def disable_fallback(self):
            self.fallback_disabled = True

        def run(self, outputs, inputs):
            self.runs.append(inputs["image"].shape)
            if self.number == failed_stage:
                raise RuntimeError("probe failure")
            return [np.zeros((1, 40, len(chars) + 2), np.float32)]

        def end_profiling(self):
            self.closed = True

    monkeypatch.setattr("onnxruntime.InferenceSession", Session)
    if failed_stage:
        with pytest.raises(RuntimeError, match="probe failure"):
            DirectMLAdapter()
        assert len(sessions) == failed_stage
    else:
        adapter = DirectMLAdapter()
        adapter.probe()
        assert len(sessions) == 4
        assert all(len(s.runs) == 2 and s.fallback_disabled for s in sessions)
        name_reader = adapter._name_ocr
        adapter.close()
        assert name_reader.session is None and adapter._engine is None
    assert all(s.closed for s in sessions)
