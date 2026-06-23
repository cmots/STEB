"""Shared text normalization for evaluation scorers."""

import re
import string
from typing import Optional

try:
    from zhon.hanzi import punctuation as zh_punctuation
except ImportError:
    zh_punctuation = ""

try:
    import zhconv
except ImportError:
    zhconv = None


_PUNCTUATION_ALL = zh_punctuation + string.punctuation
_SPECIAL_TOKENS = ("<unk>", "<pad>", "=")
_SQUARE_BRACKET_TAG_RE = re.compile(r"\s*\[[^\[\]\n]{1,80}\]\s*")


def strip_square_bracket_tags(text: Optional[str]) -> str:
    """Remove inline tags represented as square-bracket spans."""
    return _SQUARE_BRACKET_TAG_RE.sub(" ", str(text or "")).strip()


def normalize_text(
    text: Optional[str],
    lang: str,
    *,
    strip_punctuation: bool,
) -> str:
    """Normalize text before metric computation.

    The pipeline uses the same core cleanup for BLEU and COMET-family metrics:
    trim special tokens, normalize whitespace, lowercase English, and unify
    Chinese script. BLEU additionally strips punctuation to preserve the
    established scoring behavior.
    """

    normalized = strip_square_bracket_tags(text)
    for token in _SPECIAL_TOKENS:
        normalized = normalized.replace(token, " ")

    if strip_punctuation:
        for punct in _PUNCTUATION_ALL:
            if punct == "'":
                continue
            normalized = normalized.replace(punct, "")

    normalized = " ".join(normalized.strip().split())

    if lang == "zh":
        if zhconv is not None:
            normalized = zhconv.convert(normalized, "zh-cn")
        normalized = "".join(normalized.split())
    elif lang == "en":
        normalized = normalized.lower()

    return normalized
