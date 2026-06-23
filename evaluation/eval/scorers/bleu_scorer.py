"""BLEU scorer — sentence-level and corpus-level."""

from typing import Any, Dict, List

import sacrebleu

from .base import BaseScorer, EvalRecord
from .text_normalization import normalize_text


class BLEUScorer(BaseScorer):
    name = "bleu"
    requires_audio = False
    requires_reference = True

    def __init__(self, asr_only: bool = False) -> None:
        self._asr_only = asr_only

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for rec in records:
            out: Dict[str, Any] = {"id": rec.id}
            if not self.can_score(rec):
                results.append(out)
                continue

            ref = normalize_text(
                rec.ref_translation.get(rec.tgt_lang, ""),
                rec.tgt_lang,
                strip_punctuation=True,
            )
            if not ref:
                results.append(out)
                continue

            tokenize = "zh" if rec.tgt_lang == "zh" else "13a"

            if not self._asr_only:
                # Text-derived BLEU
                hyp_translation = normalize_text(
                    rec.hyp_translation,
                    rec.tgt_lang,
                    strip_punctuation=True,
                )
                if hyp_translation:
                    bleu = sacrebleu.sentence_bleu(
                        hyp_translation, [ref], tokenize=tokenize
                    )
                    out["bleu"] = float(bleu.score)

            # ASR-derived BLEU (if audio was processed)
            hyp_asr_text = normalize_text(
                rec.hyp_asr_text,
                rec.tgt_lang,
                strip_punctuation=True,
            )
            if hyp_asr_text:
                bleu_asr = sacrebleu.sentence_bleu(
                    hyp_asr_text, [ref], tokenize=tokenize
                )
                out["bleu_asr"] = float(bleu_asr.score)

            results.append(out)
        return results

    def corpus_bleu(self, records: List[EvalRecord]) -> Dict[str, float]:
        """Compute corpus-level BLEU over all samples with references.

        Missing hypotheses are kept as empty strings so benchmark-missing or
        metric-missing samples contribute the minimum BLEU instead of being
        silently dropped from the corpus aggregate.
        """
        hyps, refs = [], []
        hyps_asr, refs_asr = [], []
        for rec in records:
            ref = normalize_text(
                rec.ref_translation.get(rec.tgt_lang, ""),
                rec.tgt_lang,
                strip_punctuation=True,
            )
            if not ref:
                continue
            if not self._asr_only:
                hyp_translation = normalize_text(
                    rec.hyp_translation,
                    rec.tgt_lang,
                    strip_punctuation=True,
                )
                hyps.append(hyp_translation)
                refs.append(ref)
            hyp_asr_text = normalize_text(
                rec.hyp_asr_text,
                rec.tgt_lang,
                strip_punctuation=True,
            )
            hyps_asr.append(hyp_asr_text)
            refs_asr.append(ref)

        tokenize = "zh" if (records and records[0].tgt_lang == "zh") else "13a"
        result: Dict[str, float] = {}
        if hyps:
            result["bleu_corpus"] = float(
                sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize).score
            )
        if hyps_asr:
            result["bleu_asr_corpus"] = float(
                sacrebleu.corpus_bleu(hyps_asr, [refs_asr], tokenize=tokenize).score
            )
        return result
