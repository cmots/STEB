"""Repeated-run scorer wrapper for stochastic metrics.

Supports median, mean, majority, and robust aggregation.
Early stopping skips remaining runs when a majority score is already decided.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional

from .base import BaseScorer, EvalRecord


SCORE_FIELDS = ("emotion_score", "style_score", "event_score")
REASON_FIELDS = ("emotion_reason", "style_reason", "event_reason")


def _aggregate_robust(scores: List[float | int]) -> Optional[float | int]:
    """Aggregate three-run judge scores with a robust rule.

    Rule:
    - If two scores agree and the third differs by at least 2, keep the agreed
      score.
    - If all three scores differ, use the median.
    - Otherwise, fall back to the mean of the collected scores.

    For non-three-run inputs, fall back to the median to keep the behavior
    stable when a run is missing.
    """
    if not scores:
        return None
    if len(scores) != 3:
        return statistics.median(scores)

    ordered = sorted(scores)
    if ordered[0] == ordered[2]:
        return ordered[1]
    if ordered[0] == ordered[1] or ordered[1] == ordered[2]:
        duplicate = ordered[1]
        other = ordered[2] if ordered[0] == ordered[1] else ordered[0]
        if abs(float(other) - float(duplicate)) >= 2:
            return duplicate
        return sum(scores) / len(scores)
    return ordered[1]


def aggregate_scores(
    scores: List[float | int],
    strategy: str,
) -> Optional[float | int]:
    """Aggregate a list of scores using the given strategy.
    Returns None if scores is empty.
    """
    if not scores:
        return None
    if strategy == "median":
        return statistics.median(scores)
    if strategy == "mean":
        return sum(scores) / len(scores)
    if strategy == "majority":
        counts = Counter(scores)
        max_count = max(counts.values())
        modes = [s for s, c in counts.items() if c == max_count]
        if len(modes) == 1:
            return modes[0]
        return statistics.median(scores)  # tie → median fallback
    if strategy == "robust":
        return _aggregate_robust(scores)
    raise ValueError(f"Unknown aggregation strategy: {strategy}")


def _should_early_stop(collected: List[float | int], n_runs: int) -> bool:
    """Return True if majority is already decided."""
    if not collected:
        return False
    threshold = math.ceil(n_runs / 2)
    counts = Counter(collected)
    return any(c >= threshold for c in counts.values())


class EnsembleScorer(BaseScorer):
    """Wraps any BaseScorer to run N scoring passes and aggregate results."""
    requires_audio = False
    requires_reference = False

    def __init__(
        self,
        base_scorer: BaseScorer,
        n_runs: int = 3,
        strategy: str = "median",
        temperatures: Optional[List[float]] = None,
    ) -> None:
        self._base = base_scorer
        self._n_runs = n_runs
        self._strategy = strategy
        self._temperatures = temperatures
        self.requires_audio = getattr(base_scorer, "requires_audio", False)
        self.requires_reference = getattr(base_scorer, "requires_reference", False)

    @property
    def name(self) -> str:
        return f"{getattr(self._base, 'name', self._base.__class__.__name__)}_ensemble{self._n_runs}"

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        all_rounds: List[List[Dict[str, Any]]] = []
        for run_idx in range(self._n_runs):
            round_results = self._base.score(records)
            all_rounds.append(round_results)
            # Check early stopping
            if (
                self._strategy in {"median", "majority"}
                and run_idx >= 1
                and self._check_all_decided(all_rounds, len(records))
            ):
                break

        # Aggregate
        results: List[Dict[str, Any]] = [{"id": rec.id} for rec in records]
        for idx in range(len(records)):
            for score_field, reason_field in zip(SCORE_FIELDS, REASON_FIELDS):
                raw_scores = []
                raw_reasons = []
                for round_results in all_rounds:
                    if idx < len(round_results):
                        val = round_results[idx].get(score_field)
                        if val is not None:
                            raw_scores.append(val)
                            raw_reasons.append(round_results[idx].get(reason_field))
                if not raw_scores:
                    continue
                agg = aggregate_scores(raw_scores, self._strategy)
                if agg is not None:
                    results[idx][score_field] = agg
                    dim_prefix = score_field.replace("_score", "")
                    results[idx][f"{dim_prefix}_scores_raw"] = raw_scores
                    results[idx][f"{dim_prefix}_score_std"] = (
                        statistics.stdev(raw_scores) if len(raw_scores) > 1 else 0.0
                    )
                    best_idx = min(
                        range(len(raw_scores)),
                        key=lambda i: abs(raw_scores[i] - agg),
                    )
                    results[idx][reason_field] = raw_reasons[best_idx]
        return results

    def _check_all_decided(self, all_rounds, n_records):
        for idx in range(n_records):
            for score_field in SCORE_FIELDS:
                collected = []
                for round_results in all_rounds:
                    if idx < len(round_results):
                        val = round_results[idx].get(score_field)
                        if val is not None:
                            collected.append(val)
                if collected and not _should_early_stop(collected, self._n_runs):
                    return False
        return True
