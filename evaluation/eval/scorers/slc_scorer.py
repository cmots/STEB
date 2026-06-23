"""Speech Length Consistency scorer — multi-threshold boolean pass/fail."""

from typing import Any, Dict, List, Optional

from .base import BaseScorer, EvalRecord


class SLCScorer(BaseScorer):
    name = "slc"
    requires_audio = True
    requires_reference = False

    def __init__(self, thresholds: Optional[List[float]] = None) -> None:
        self._thresholds = thresholds or [0.2, 0.4]

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        """Placeholder — SLC depends on duration_ratio from DurationScorer.

        Use compute_from_duration() instead, called by the orchestrator.
        """
        return [{"id": rec.id} for rec in records]

    def compute_from_duration(
        self, duration_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compute SLC from pre-computed duration ratios."""
        results: List[Dict[str, Any]] = []
        for dr in duration_results:
            out: Dict[str, Any] = {"id": dr.get("id", "")}
            ratio = dr.get("duration_ratio")
            if ratio is not None and ratio > 0:
                for threshold in self._thresholds:
                    key = f"slc_{threshold}"
                    out[key] = bool((1 - threshold) < ratio < (1 + threshold))
            results.append(out)
        return results
