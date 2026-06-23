"""Base scorer interface for all evaluation metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalRecord:
    """Single evaluation record joining benchmark + results data."""

    id: str
    src_lang: str
    tgt_lang: str

    # Reference fields (from benchmark JSONL)
    ref_text: str = ""
    ref_text_with_events: str = ""
    src_text_with_event: str = ""
    ref_translation: Dict[str, str] = field(default_factory=dict)
    ref_emotion: str = ""
    ref_style: str = ""
    ref_caption: str = ""
    ref_wav_path: Optional[str] = None

    # Hypothesis fields (from results JSONL)
    hyp_text: Optional[str] = None
    hyp_translation: str = ""
    hyp_wav_path: Optional[str] = None
    model_name: str = ""
    error: Optional[str] = None

    # Text-derived fields (parsed from hyp_text CoT block)
    hyp_emotion_text: Optional[str] = None
    hyp_style_text: Optional[str] = None
    hyp_transcription_text: Optional[str] = None

    # Audio-derived features (filled after Phase 2)
    hyp_asr_text: Optional[str] = None
    hyp_caption: Optional[str] = None
    hyp_emotion: Optional[str] = None
    hyp_style: Optional[str] = None
    hyp_text_with_events: Optional[str] = None
    hyp_asr_text_with_event: Optional[str] = None
    hyp_timestamp: Optional[str] = None


class BaseScorer(ABC):
    """Base class for all evaluation scorers."""

    name: str = ""
    requires_audio: bool = False
    requires_reference: bool = False

    @abstractmethod
    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        """Score a batch of records.

        Returns a list of length len(records), positionally aligned.
        Each dict contains {field_name: value} pairs to merge into output.
        For records where can_score() returns False, return {"id": record.id}
        (id only, no metric fields).
        Every dict MUST include an "id" field for validation.
        """
        ...

    def can_score(self, record: EvalRecord) -> bool:
        """Check if this scorer can run on the given record."""
        if record.error is not None:
            return False
        if self.requires_audio and record.hyp_wav_path is None:
            return False
        if self.requires_reference:
            ref = record.ref_translation.get(record.tgt_lang)
            if not ref:
                return False
        return True
