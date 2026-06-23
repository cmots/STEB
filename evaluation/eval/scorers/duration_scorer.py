"""Duration scorer — computes hyp/ref duration ratio."""

import os
from typing import Any, Dict, List

import soundfile as sf
import torch
import torchaudio

from .base import BaseScorer, EvalRecord


def audio_duration(path: str, target_sr: int = 16000) -> float:
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = torchaudio.functional.resample(torch.tensor(data), sr, target_sr).numpy()
    return float(len(data) / target_sr)


class DurationScorer(BaseScorer):
    name = "duration"
    requires_audio = True
    requires_reference = False

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for rec in records:
            out: Dict[str, Any] = {"id": rec.id}
            if not self.can_score(rec):
                results.append(out)
                continue
            try:
                ref_path = rec.ref_wav_path
                hyp_path = rec.hyp_wav_path
                if not ref_path or not hyp_path:
                    results.append(out)
                    continue
                if not os.path.exists(ref_path) or not os.path.exists(hyp_path):
                    results.append(out)
                    continue
                ref_dur = audio_duration(ref_path)
                hyp_dur = audio_duration(hyp_path)
                ratio = hyp_dur / ref_dur if ref_dur > 0 else 0.0
                out["duration_ratio"] = ratio
            except Exception as e:
                print(f"[WARN] Duration scoring failed for {rec.id}: {e}")
            results.append(out)
        return results
