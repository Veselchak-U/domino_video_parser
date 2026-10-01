"""Local Cyrillic-first name OCR retaining individual CTC probabilities."""

import hashlib
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .ocr_result import NAME_OCR_THRESHOLD, OCRLine, OCRResult

MODEL_HASH = "1efb65bdc460af1c0e8733d005b20952b17ca5aac10ddb56c968333791c5eaa3"
MODEL_FILE = "cyrillic_PP-OCRv3_rec_mobile.onnx"


@dataclass(frozen=True)
class NameSymbol:
    text: str
    confidence: float
    start: float
    end: float


@dataclass(frozen=True)
class NameReading(OCRResult):
    symbols: tuple[NameSymbol, ...] = ()
    alternatives: tuple[dict, ...] = ()

    @property
    def text(self):
        return "".join(s.text if s.confidence > NAME_OCR_THRESHOLD else "*" for s in self.symbols)

    @property
    def reason(self):
        if not any(
            s.confidence > NAME_OCR_THRESHOLD and not s.text.isspace() for s in self.symbols
        ):
            return "low_confidence" if self.symbols else "no_text"
        return (
            "partial_name"
            if any(s.confidence <= NAME_OCR_THRESHOLD for s in self.symbols)
            else None
        )


class CTCDecoder:
    def decode(self, predictions, characters, width, content_ratio=1.0):
        scores = predictions[0]
        if scores.shape[1] != len(characters):
            raise RuntimeError("Словарь модели имён не соответствует выходу OCR")
        indices, probabilities = scores.argmax(axis=1), scores.max(axis=1)
        tokens = []
        for i, token in enumerate(indices):
            if token == 0 or (i and token == indices[i - 1]):
                continue
            tokens.append((i, str(characters[token]), float(probabilities[i])))
        if not tokens:
            return ()
        # Midpoints bound decoded intervals, including blank gaps. They describe
        # token alignment, not inferred missing characters or exact glyph boxes.
        step = width / (len(indices) * content_ratio)
        centers = [(i + 0.5) * step for i, _, _ in tokens]
        result = []
        for n, (_, text, probability) in enumerate(tokens):
            left = (centers[n - 1] + centers[n]) / 2 if n else max(0, centers[n] - step)
            right = (centers[n] + centers[n + 1]) / 2 if n + 1 < len(tokens) else centers[n] + step
            if left < width:
                result.append(NameSymbol(text, probability, left, min(width, right)))
        return tuple(result)


class NameOCR:
    def __init__(self, engine, session=None):
        import onnxruntime as ort

        self.engine = engine
        path = self.model_path()
        if session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        self.session = session
        chars = session.get_modelmeta().custom_metadata_map.get("character")
        if not chars:
            raise RuntimeError("Модель имён не содержит словаря символов")
        self.characters = ["blank", *chars.splitlines(), " "]
        if session.get_outputs()[0].shape[-1] != len(self.characters):
            raise RuntimeError("Словарь модели имён не соответствует выходу OCR")
        required = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        if not set(required).issubset(self.characters):
            raise RuntimeError("Модель имён не содержит полного русского алфавита")
        self.decoder = CTCDecoder()

    @staticmethod
    def model_path(path=None):
        path = Path(path) if path is not None else Path(__file__).parent / "models" / MODEL_FILE
        try:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            raise RuntimeError(f"Обязательная локальная модель имён недоступна: {path}") from error
        if digest != MODEL_HASH:
            raise RuntimeError(f"Повреждена локальная модель имён (SHA-256): {path}")
        return path

    def _predict(self, image, session, characters, height, minimum_width):
        h, w = image.shape[:2]
        resized = math.ceil(height * w / h)
        width = max(minimum_width, resized)
        normalized = cv2.resize(image, (resized, height)).astype(np.float32)
        normalized = (normalized.transpose(2, 0, 1) / 255 - 0.5) / 0.5
        batch = np.zeros((1, 3, height, width), dtype=np.float32)
        batch[0, :, :, :resized] = normalized
        predictions = session.run(None, {session.get_inputs()[0].name: batch})[0]
        return self.decoder.decode(predictions, characters, w, resized / width)

    def read(self, crop):
        boxes, _ = self.engine(crop, use_rec=False, use_cls=False)
        if not boxes:
            return NameReading()
        regions = self.engine.get_crop_img_list(crop, np.asarray(boxes, dtype=np.float32))
        symbols, rows, alternatives = [], [], []
        secondary = self.engine.text_rec
        for box, image in zip(boxes, regions):
            primary = self._predict(image, self.session, self.characters, 48, 320)
            other = self._predict(
                image,
                secondary.session.session,
                secondary.postprocess_op.character,
                secondary.rec_image_shape[1],
                secondary.rec_image_shape[2],
            )
            for source, tokens in (("cyrillic", primary), ("supplementary", other)):
                text = "".join(s.text for s in tokens)
                confidence = sum(s.confidence for s in tokens) / len(tokens) if tokens else 0
                if tokens:
                    rows.append(OCRLine(text, confidence))
                alternatives.append(
                    dict(model=source, box=box, symbols=[asdict(s) for s in tokens])
                )
            merged = self.merge(primary, other)
            left = min(p[0] for p in box)
            scale = (max(p[0] for p in box) - left) / image.shape[1]
            symbols.extend(
                NameSymbol(s.text, s.confidence, left + s.start * scale, left + s.end * scale)
                for s in merged
            )
        return NameReading(tuple(rows), tuple(symbols), tuple(alternatives))

    def merge(self, primary, other):
        # Preserve confidently decoded foreign logical sequences, including RTL
        # and combining marks, rather than applying left-to-right fusion to them.
        if self._logical_sequence(primary):
            return primary
        if self._logical_sequence(other) and not any(
            s.confidence > NAME_OCR_THRESHOLD for s in primary
        ):
            return other
        result, used = [], set()
        for token in primary:
            matches = []
            for i, candidate in enumerate(other):
                overlap = max(0, min(token.end, candidate.end) - max(token.start, candidate.start))
                union = max(token.end, candidate.end) - min(token.start, candidate.start)
                iou = overlap / union if union else 0
                if i not in used and iou >= 0.5 and candidate.confidence > NAME_OCR_THRESHOLD:
                    matches.append((iou, candidate.confidence, -i, candidate))
            if token.confidence <= NAME_OCR_THRESHOLD and matches:
                _, _, negative_index, token = max(matches, key=lambda m: m[:3])
                used.add(-negative_index)
            result.append(token)
        for i, token in enumerate(other):
            if i not in used and not any(
                min(token.end, p.end) > max(token.start, p.start) for p in primary
            ):
                result.append(token)
        return tuple(sorted(result, key=lambda s: s.start))

    def _logical_sequence(self, tokens):
        return any(
            s.confidence > NAME_OCR_THRESHOLD
            and any(
                unicodedata.category(c).startswith("M")
                or (c.isalpha() and not (c.isascii() or "А" <= c <= "я" or c in "Ёё"))
                for c in s.text
            )
            for s in tokens
        )

    def close(self):
        self.session = None
        self.engine = None
