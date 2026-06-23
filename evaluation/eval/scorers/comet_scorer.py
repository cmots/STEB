"""COMET scorer — supports local checkpoints, wmt22-comet-da, cometkiwi, and XCOMET models."""

import os
from typing import Any, Dict, List, Optional

from .base import BaseScorer, EvalRecord
from .text_normalization import normalize_text

try:
    from comet import download_model as _download_model, load_from_checkpoint as _load_from_checkpoint
except ImportError:
    _download_model = None
    _load_from_checkpoint = None


def _is_xcomet(model_name: str) -> bool:
    return "xcomet" in model_name.lower()


def _resolve_local_checkpoint(model_name: str) -> Optional[str]:
    if not model_name:
        return None

    expanded = os.path.expanduser(model_name)
    if os.path.isfile(expanded) and expanded.endswith(".ckpt"):
        return expanded

    candidates = [
        os.path.join(expanded, "checkpoints", "model.ckpt"),
    ]

    if os.path.isdir(expanded):
        snapshots_dir = os.path.join(expanded, "snapshots")
        if os.path.isdir(snapshots_dir):
            for snapshot in sorted(os.listdir(snapshots_dir)):
                candidates.append(
                    os.path.join(snapshots_dir, snapshot, "checkpoints", "model.ckpt")
                )
        models_dirs = [d for d in os.listdir(expanded) if d.startswith("models--")] if os.path.isdir(expanded) else []
        for model_dir in sorted(models_dirs):
            nested_snapshots = os.path.join(expanded, model_dir, "snapshots")
            if os.path.isdir(nested_snapshots):
                for snapshot in sorted(os.listdir(nested_snapshots)):
                    candidates.append(
                        os.path.join(
                            nested_snapshots, snapshot, "checkpoints", "model.ckpt"
                        )
                    )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_model(model_name: str) -> Any:
    """Download (if needed) and load a COMET model."""
    if _load_from_checkpoint is None:
        raise RuntimeError("unbabel-comet not installed. Run: pip install unbabel-comet")
    local_checkpoint = _resolve_local_checkpoint(model_name)
    if local_checkpoint is not None:
        return _load_from_checkpoint(local_checkpoint)
    if _download_model is None:
        raise RuntimeError("unbabel-comet download support is unavailable.")
    model_path = _download_model(model_name)
    return _load_from_checkpoint(model_path)


class COMETScorer(BaseScorer):
    name = "comet"
    requires_audio = False
    requires_reference = False

    def __init__(
        self,
        ref_model_name: str = "Unbabel/wmt22-comet-da",
        qe_model_name: Optional[str] = "Unbabel/wmt22-cometkiwi-da",
        batch_size: int = 8,
        gpus: int = 1,
        field_prefix: str = "comet",
        asr_only: bool = False,
    ) -> None:
        self._ref_model_name = ref_model_name
        self._qe_model_name = qe_model_name
        self._batch_size = batch_size
        self._gpus = gpus
        self._field_prefix = field_prefix
        self._asr_only = asr_only

        self._is_unified = _is_xcomet(ref_model_name)

        self._ref_model: Any = None
        self._qe_model: Any = None

    def _ensure_models(self):
        if self._ref_model is None:
            print(f"[COMET] Loading ref model: {self._ref_model_name}")
            self._ref_model = _load_model(self._ref_model_name)
        if not self._is_unified and self._qe_model is None and self._qe_model_name:
            print(f"[COMET] Loading QE model: {self._qe_model_name}")
            self._qe_model = _load_model(self._qe_model_name)

    def _normalize_source(self, text: Optional[str], lang: str) -> str:
        return normalize_text(text, lang, strip_punctuation=False)

    def _normalize_target(self, text: Optional[str], lang: str) -> str:
        return normalize_text(text, lang, strip_punctuation=False)

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        self._ensure_models()
        results: List[Dict[str, Any]] = [{"id": rec.id} for rec in records]

        if not records:
            print(f"[COMET] No records to score for {self._field_prefix} in this shard.")
            return results

        qe_text_data = []
        qe_text_indices = []
        ref_text_data = []
        ref_text_indices = []
        qe_asr_data = []
        qe_asr_indices = []
        ref_asr_data = []
        ref_asr_indices = []

        for idx, rec in enumerate(records):
            if rec.error is not None:
                continue
            source = self._normalize_source(rec.ref_text, rec.src_lang)
            if not source:
                continue

            hyp_translation = self._normalize_target(rec.hyp_translation, rec.tgt_lang)
            hyp_asr_text = self._normalize_target(rec.hyp_asr_text, rec.tgt_lang)
            ref_trans = self._normalize_target(
                rec.ref_translation.get(rec.tgt_lang, ""),
                rec.tgt_lang,
            )

            if hyp_translation and not self._asr_only:
                qe_text_data.append({"src": source, "mt": hyp_translation})
                qe_text_indices.append(idx)
                if ref_trans:
                    ref_text_data.append({
                        "src": source,
                        "mt": hyp_translation,
                        "ref": ref_trans,
                    })
                    ref_text_indices.append(idx)

            if hyp_asr_text:
                qe_asr_data.append({"src": source, "mt": hyp_asr_text})
                qe_asr_indices.append(idx)
                if ref_trans:
                    ref_asr_data.append({
                        "src": source,
                        "mt": hyp_asr_text,
                        "ref": ref_trans,
                    })
                    ref_asr_indices.append(idx)

        if qe_text_data and (self._is_unified or self._qe_model is not None):
            qe_model = self._ref_model if self._is_unified else self._qe_model
            qe_output = qe_model.predict(
                qe_text_data, batch_size=self._batch_size, gpus=self._gpus
            )
            for score_val, idx in zip(qe_output.scores, qe_text_indices):
                results[idx][f"{self._field_prefix}_qe"] = float(score_val)

        if ref_text_data:
            ref_output = self._ref_model.predict(
                ref_text_data, batch_size=self._batch_size, gpus=self._gpus
            )
            for score_val, idx in zip(ref_output.scores, ref_text_indices):
                results[idx][f"{self._field_prefix}_ref"] = float(score_val)

        if qe_asr_data and (self._is_unified or self._qe_model is not None):
            qe_model = self._ref_model if self._is_unified else self._qe_model
            qe_output = qe_model.predict(
                qe_asr_data, batch_size=self._batch_size, gpus=self._gpus
            )
            for score_val, idx in zip(qe_output.scores, qe_asr_indices):
                results[idx][f"{self._field_prefix}_qe_asr"] = float(score_val)

        if ref_asr_data:
            ref_output = self._ref_model.predict(
                ref_asr_data, batch_size=self._batch_size, gpus=self._gpus
            )
            for score_val, idx in zip(ref_output.scores, ref_asr_indices):
                results[idx][f"{self._field_prefix}_ref_asr"] = float(score_val)

        return results
