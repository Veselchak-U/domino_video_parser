"""OCR evidence before confidence filtering, shared by CPU and GPU."""

from dataclasses import dataclass

OCR_THRESHOLD = 0.8
NAME_OCR_THRESHOLD = 0.7


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    rows: tuple[OCRLine, ...] = ()

    @classmethod
    def from_rows(cls, rows):
        return cls(tuple(OCRLine(row[1], float(row[2])) for row in (rows or [])))

    @property
    def text(self):
        return " ".join(row.text for row in self.rows if row.confidence > OCR_THRESHOLD)

    @property
    def reason(self):
        if self.text.strip():
            return None
        return "low_confidence" if any(row.text.strip() for row in self.rows) else "no_text"
