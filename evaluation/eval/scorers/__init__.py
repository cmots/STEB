"""Evaluation scorer modules.

Heavy scorer classes (COMET, SpeakerSim, etc.) are imported lazily
to avoid requiring torch/torchaudio at package import time.
"""

from .base import BaseScorer, EvalRecord

__all__ = [
    "BaseScorer", "EvalRecord",
    "BLEUScorer", "COMETScorer", "EventF1Scorer",
    "DurationScorer", "SpeakerSimScorer", "SLCScorer",
    "LLMEmotionScorer", "LLMStyleScorer", "LLMEventScorer",
]

# Lazy imports for modules with heavy dependencies (torch, torchaudio, etc.)
_LAZY_IMPORTS = {
    "BLEUScorer": ".bleu_scorer",
    "COMETScorer": ".comet_scorer",
    "EventF1Scorer": ".event_f1_scorer",
    "DurationScorer": ".duration_scorer",
    "SpeakerSimScorer": ".speaker_sim_scorer",
    "SLCScorer": ".slc_scorer",
    "LLMEmotionScorer": ".llm_emotion_scorer",
    "LLMStyleScorer": ".llm_style_scorer",
    "LLMEventScorer": ".llm_event_scorer",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        cls = getattr(module, name)
        globals()[name] = cls  # cache for subsequent access
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
