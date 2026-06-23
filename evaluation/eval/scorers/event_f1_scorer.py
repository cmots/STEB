"""Sound event tag F1 scorer — precision, recall, F1 on [Tag] markers."""

import re
from collections import Counter
from typing import Any, Dict, List

from .base import BaseScorer, EvalRecord

KNOWN_TAGS = {
    "[Breathing]", "[Cough]", "[Laughter]", "[Sneeze]",
    "[Crying]", "[Whispering]", "[Sigh]", "[Pant]", "[Burp]",
}


def extract_tags(text: str) -> List[str]:
    """Extract known [Tag] markers from text."""
    if not text:
        return []
    return [t for t in re.findall(r"\[[A-Za-z]+\]", text) if t in KNOWN_TAGS]


def multiset_prf(ref_tags: List[str], hyp_tags: List[str]) -> Dict[str, float]:
    """Compute precision, recall, F1 on multisets of tags."""
    if not ref_tags and not hyp_tags:
        return {"event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0}
    if not ref_tags:
        return {"event_precision": 0.0, "event_recall": 1.0, "event_f1": 0.0}
    if not hyp_tags:
        return {"event_precision": 1.0, "event_recall": 0.0, "event_f1": 0.0}

    ref_counter = Counter(ref_tags)
    hyp_counter = Counter(hyp_tags)

    tp = sum((ref_counter & hyp_counter).values())
    precision = tp / sum(hyp_counter.values()) if sum(hyp_counter.values()) > 0 else 0.0
    recall = tp / sum(ref_counter.values()) if sum(ref_counter.values()) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"event_precision": precision, "event_recall": recall, "event_f1": f1}


class EventF1Scorer(BaseScorer):
    name = "event_f1"
    requires_audio = False
    requires_reference = False

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for rec in records:
            out: Dict[str, Any] = {"id": rec.id}
            if rec.error is not None:
                results.append(out)
                continue

            ref_tags = extract_tags(rec.src_text_with_event or rec.ref_text_with_events)
            hyp_text = rec.hyp_asr_text_with_event
            hyp_tags = extract_tags(hyp_text)

            out.update(multiset_prf(ref_tags, hyp_tags))
            results.append(out)
        return results
