"""Speaker similarity scorer — ECAPA-TDNN + WavLM-Large (Seed-TTS-eval / UniSpeech).

This scorer runs inside an isolated Python environment (s3prl 0.3.1 +
fairseq 0.12.2 + torch 1.9) invoked via $SPEAKER_SIM_PYTHON.  It must
never be imported from the main eval environment.

Core inference is delegated to the vendored verification() function from
  evaluation/eval/sim/thirdparty_unispeech/downstreams/speaker_verification/
which is the byte-for-byte same code as Seed-TTS-eval.  No batching, no
padding — pair-by-pair forward matches the reference implementation exactly.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from tqdm import tqdm

try:
    from .base import BaseScorer, EvalRecord
except ImportError:  # when invoked as a subprocess entry point
    from scorers.base import BaseScorer, EvalRecord


# ---------------------------------------------------------------------------
# Vendored path — appended lazily in _ensure_model so the main env never
# pays the import cost even though the file lives in the scorers/ package.
# ---------------------------------------------------------------------------
_VENDOR_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "sim", "thirdparty_unispeech",
        "downstreams", "speaker_verification",
    )
)


def _init_model(model_name: str, checkpoint_path: str):
    """Load ECAPA-TDNN + WavLM via vendored init_model()."""
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    from verification import init_model  # noqa: PLC0415 (lazy import by design)
    return init_model(model_name, checkpoint_path)


def _call_verification(
    model_name: str,
    hyp_wav_path: str,
    ref_wav_path: str,
    checkpoint_path: str,
    model,
    device: str,
):
    """Single-pair speaker similarity via vendored verification().

    Returns (sim_tensor, model) — the model is returned so the caller can
    reuse it across pairs without reloading (Seed-TTS-eval pattern).
    """
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    from verification import verification  # noqa: PLC0415
    return verification(
        model_name,
        hyp_wav_path,
        ref_wav_path,
        use_gpu=True,
        checkpoint=checkpoint_path,
        wav1_start_sr=0,
        wav2_start_sr=0,
        wav1_end_sr=-1,
        wav2_end_sr=-1,
        model=model,
        device=device,
    )


class SpeakerSimScorer(BaseScorer):
    """Speaker similarity using ECAPA-TDNN fine-tuned on VoxCeleb (UniSpeech).

    Matches Seed-TTS-eval's verification_pair_list_v2.py semantics exactly:
    pair-by-pair inference, model reuse across pairs, no batching or padding.
    """

    name = "speaker_sim"
    requires_audio = True
    requires_reference = False

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "wavlm_large",
        device: str = "cuda:0",
        partial_results_path: Optional[str] = None,
        flush_every: int = 32,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._model_name = model_name
        self._device = device
        self._partial_results_path = partial_results_path
        self._flush_every = flush_every
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            print(f"[SpeakerSim] Loading {self._model_name} + ECAPA-TDNN from "
                  f"{self._checkpoint_path} ...")
            self._model = _init_model(self._model_name, self._checkpoint_path)

    def _flush(self, results: List[Dict[str, Any]]) -> None:
        if not self._partial_results_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._partial_results_path)), exist_ok=True)
        tmp = f"{self._partial_results_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        os.replace(tmp, self._partial_results_path)

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        if not records:
            print("[SpeakerSim] No records to score in this shard.")
            return []

        self._ensure_model()

        results: List[Dict[str, Any]] = [{"id": rec.id} for rec in records]

        for idx, rec in enumerate(tqdm(records, desc="SpeakerSim")):
            if not self.can_score(rec):
                continue
            if not rec.hyp_wav_path or not rec.ref_wav_path:
                continue
            if not os.path.exists(rec.hyp_wav_path) or not os.path.exists(rec.ref_wav_path):
                continue

            try:
                sim_tensor, self._model = _call_verification(
                    self._model_name,
                    rec.hyp_wav_path,
                    rec.ref_wav_path,
                    self._checkpoint_path,
                    self._model,
                    self._device,
                )
                sim_val = float(sim_tensor.cpu().item())
                if not math.isnan(sim_val):
                    results[idx]["speaker_similarity"] = sim_val
            except Exception as e:
                print(f"[WARN] speaker_sim failed on {rec.id}: {e}")
                traceback.print_exc()
                # Leave results[idx] as {id: ...} — will get 0.0 fallback at aggregation.

            if self._partial_results_path and (idx + 1) % self._flush_every == 0:
                self._flush(results)

        self._flush(results)
        return results
