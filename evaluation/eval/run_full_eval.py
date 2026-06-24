"""Unified evaluation orchestrator — delegates to modular scorers."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

# Ensure scorers package is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scorers.base import EvalRecord
from scorers.exp_ensemble_scorer import EnsembleScorer
from shard_utils import (
    align_sparse_rows_to_records,
    build_pending_shard_records,
    collect_completed_ids_from_rows,
    load_sparse_group_rows,
)

# ---- Helpers ----

def load_eval_records(path: str) -> List[EvalRecord]:
    """Load EvalRecord list from JSONL (output of data_loader or merge_hyp_features)."""
    records: List[EvalRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(EvalRecord(**{
                    k: v for k, v in d.items() if k in EvalRecord.__dataclass_fields__
                }))
            except Exception as e:
                print(f"[WARN] Skipping record: {e}")
    return records


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_scorer_results(
    records: List[EvalRecord],
    all_results: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merge results from all scorers into a single list of dicts."""
    merged: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        row: Dict[str, Any] = {
            "id": rec.id,
            "model_name": rec.model_name,
            "src_lang": rec.src_lang,
            "tgt_lang": rec.tgt_lang,
            "has_audio": bool(rec.hyp_wav_path and os.path.exists(rec.hyp_wav_path)),
            "error": rec.error,
        }
        for scorer_results in all_results:
            if idx < len(scorer_results):
                sr = scorer_results[idx]
                # Validate id match
                if sr.get("id") != rec.id:
                    print(f"[WARN] Scorer result id mismatch at index {idx}: "
                          f"expected {rec.id}, got {sr.get('id')}")
                    continue
                for k, v in sr.items():
                    if k != "id":
                        row[k] = v
        merged.append(row)
    return merged


def merge_group_rows(
    base_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> None:
    """Merge a group-level aligned result list into accumulated rows."""
    for idx, row in enumerate(base_rows):
        if idx >= len(new_rows):
            continue
        new_row = new_rows[idx]
        if new_row.get("id") != row["id"]:
            print(
                f"[WARN] Group result id mismatch at index {idx}: "
                f"expected {row['id']}, got {new_row.get('id')}"
            )
            continue
        for key, value in new_row.items():
            if key != "id":
                row[key] = value


def get_phase3_groups(
    args: argparse.Namespace,
    allow_llm_without_url: bool = False,
) -> List[str]:
    """Return the ordered metric groups for Phase 3 subprocess isolation."""
    groups = []
    if not getattr(args, "disable_basic_audio", False):
        groups.append("bleu_audio")
    if args.enable_llm and (args.llm_url or allow_llm_without_url):
        groups.append("llm")
    if args.enable_speaker_sim:
        groups.append("speaker_sim")
    if args.enable_comet and args.base_comet_model:
        groups.append("comet")
    if args.enable_xcomet and args.xcomet_model:
        groups.append("xcomet")
    return groups


def run_metric_group_locally(
    records: List[EvalRecord],
    args: argparse.Namespace,
    group: str,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run one isolated metric group inside the current process."""
    group_rows = [{"id": rec.id} for rec in records]
    corpus_bleu: Dict[str, float] = {}

    if group == "bleu_audio":
        from scorers.bleu_scorer import BLEUScorer
        from scorers.duration_scorer import DurationScorer
        from scorers.slc_scorer import SLCScorer

        if getattr(args, "asr_basic_only", False):
            bleu_scorer = BLEUScorer(asr_only=True)
        else:
            bleu_scorer = BLEUScorer()
        print("[Phase 3] Running BLEU scorer...")
        merge_group_rows(group_rows, bleu_scorer.score(records))
        corpus_bleu = bleu_scorer.corpus_bleu(records)

        if getattr(args, "asr_basic_only", False):
            return group_rows, corpus_bleu

        print("[Phase 4] Running Duration scorer...")
        duration_results = DurationScorer().score(records)
        merge_group_rows(group_rows, duration_results)

        thresholds = [float(t) for t in args.slc_thresholds.split(",") if t]
        slc_scorer = SLCScorer(thresholds=thresholds)
        merge_group_rows(group_rows, slc_scorer.compute_from_duration(duration_results))
        return group_rows, corpus_bleu

    if group == "speaker_sim":
        from scorers.speaker_sim_scorer import SpeakerSimScorer

        print("[Phase 4] Running Speaker Similarity scorer...")
        ckpt = getattr(args, "speaker_sim_ckpt", None) or os.environ.get("SPEAKER_SIM_CKPT")
        if not ckpt or not os.path.isfile(ckpt):
            raise RuntimeError(
                f"SPEAKER_SIM_CKPT is not set or file not found: {ckpt!r}. "
                "Set SPEAKER_SIM_CKPT to wavlm_large_finetune.pth; see README.md."
            )
        scorer_kwargs: Dict[str, Any] = {"checkpoint_path": ckpt}
        partial_results_path = getattr(args, "partial_results_path", None)
        if partial_results_path:
            scorer_kwargs["partial_results_path"] = partial_results_path
        merge_group_rows(
            group_rows,
            SpeakerSimScorer(**scorer_kwargs).score(records),
        )
        return group_rows, corpus_bleu

    if group == "comet":
        from scorers.comet_scorer import COMETScorer

        print("[Phase 3] Running base COMET scorer...")
        scorer = COMETScorer(
            ref_model_name=args.base_comet_model,
            qe_model_name=args.comet_qe_model,
            batch_size=64,
            field_prefix="comet",
            asr_only=getattr(args, "asr_basic_only", False),
        )
        merge_group_rows(group_rows, scorer.score(records))
        return group_rows, corpus_bleu

    if group == "xcomet":
        from scorers.comet_scorer import COMETScorer

        print("[Phase 3] Running XCOMET scorer...")
        scorer = COMETScorer(
            ref_model_name=args.xcomet_model,
            qe_model_name=None,
            batch_size=64,
            field_prefix="xcomet",
            asr_only=getattr(args, "asr_basic_only", False),
        )
        merge_group_rows(group_rows, scorer.score(records))
        return group_rows, corpus_bleu

    if group == "llm":
        llm_model = args.llm_model or "Qwen/Qwen3-30B-A3B-Instruct"
        llm_conc = args.llm_concurrency
        llm_runs = int(getattr(args, "llm_ensemble_runs", 3))
        llm_strategy = getattr(args, "llm_ensemble_strategy", "robust")
        llm_prompt_version = getattr(args, "llm_prompt_version", "default")

        if not getattr(args, "disable_llm_emotion", False):
            from scorers.llm_emotion_scorer import LLMEmotionScorer

            print(
                f"[Phase 5] Running LLM Emotion scorer "
                f"({llm_runs} repeated run(s), aggregation={llm_strategy})..."
            )
            scorer = EnsembleScorer(
                LLMEmotionScorer(
                    args.llm_url,
                    llm_model,
                    llm_conc,
                    prompt_version=llm_prompt_version,
                ),
                n_runs=llm_runs,
                strategy=llm_strategy,
            )
            merge_group_rows(group_rows, scorer.score(records))

        if not getattr(args, "disable_llm_style", False):
            from scorers.llm_style_scorer import LLMStyleScorer

            print(
                f"[Phase 5] Running LLM Style scorer "
                f"({llm_runs} repeated run(s), aggregation={llm_strategy})..."
            )
            scorer = EnsembleScorer(
                LLMStyleScorer(
                    args.llm_url,
                    llm_model,
                    llm_conc,
                    prompt_version=llm_prompt_version,
                ),
                n_runs=llm_runs,
                strategy=llm_strategy,
            )
            merge_group_rows(group_rows, scorer.score(records))

        if not getattr(args, "disable_llm_event", False):
            from scorers.llm_event_scorer import LLMEventScorer

            print(
                f"[Phase 5] Running LLM Event scorer "
                f"({llm_runs} repeated run(s), aggregation={llm_strategy})..."
            )
            scorer = EnsembleScorer(
                LLMEventScorer(
                    args.llm_url,
                    llm_model,
                    llm_conc,
                    prompt_version=llm_prompt_version,
                ),
                n_runs=llm_runs,
                strategy=llm_strategy,
            )
            merge_group_rows(group_rows, scorer.score(records))
        return group_rows, corpus_bleu

    raise ValueError(f"Unknown score group: {group}")


def build_subprocess_command(
    args: argparse.Namespace,
    group: str,
    partial_results_path: str,
    partial_corpus_path: str,
) -> List[str]:
    """Build a self-invocation command for one isolated metric group."""
    if group == "speaker_sim":
        python_exec = getattr(args, "speaker_sim_python", None) or os.environ.get("SPEAKER_SIM_PYTHON")
        if not python_exec or not os.path.isfile(python_exec):
            raise RuntimeError(
                f"SPEAKER_SIM_PYTHON is not set or not found: {python_exec!r}. "
                "Create the speaker-sim uv environment and set SPEAKER_SIM_PYTHON; see README.md."
            )
    else:
        python_exec = sys.executable
    cmd = [
        python_exec,
        os.path.abspath(__file__),
        "--input",
        args.input,
        "--output_dir",
        args.output_dir,
        "--src_lang",
        args.src_lang,
        "--tgt_lang",
        args.tgt_lang,
        "--slc_thresholds",
        args.slc_thresholds,
        "--score_group",
        group,
        "--partial_results_path",
        partial_results_path,
        "--partial_corpus_path",
        partial_corpus_path,
    ]

    if args.base_comet_model:
        cmd.extend(["--base_comet_model", args.base_comet_model])
    if args.xcomet_model:
        cmd.extend(["--xcomet_model", args.xcomet_model])
    if args.comet_qe_model:
        cmd.extend(["--comet_qe_model", args.comet_qe_model])
    if args.enable_comet:
        cmd.append("--enable_comet")
    if args.enable_xcomet:
        cmd.append("--enable_xcomet")
    if args.enable_speaker_sim:
        cmd.append("--enable_speaker_sim")
    if getattr(args, "speaker_sim_ckpt", None):
        cmd.extend(["--speaker_sim_ckpt", args.speaker_sim_ckpt])
    if getattr(args, "speaker_sim_python", None):
        cmd.extend(["--speaker_sim_python", args.speaker_sim_python])
    if args.enable_llm:
        cmd.append("--enable_llm")
    if getattr(args, "disable_llm_emotion", False):
        cmd.append("--disable_llm_emotion")
    if getattr(args, "disable_llm_style", False):
        cmd.append("--disable_llm_style")
    if getattr(args, "disable_llm_event", False):
        cmd.append("--disable_llm_event")
    if getattr(args, "asr_basic_only", False):
        cmd.append("--asr_basic_only")
    if getattr(args, "disable_basic_audio", False):
        cmd.append("--disable_basic_audio")
    if args.llm_url:
        cmd.extend(["--llm_url", args.llm_url])
    if args.llm_model:
        cmd.extend(["--llm_model", args.llm_model])
    cmd.extend(["--llm_concurrency", str(args.llm_concurrency)])
    cmd.extend(["--llm_ensemble_runs", str(getattr(args, "llm_ensemble_runs", 3))])
    cmd.extend(["--llm_ensemble_strategy", getattr(args, "llm_ensemble_strategy", "robust")])
    cmd.extend(["--llm_prompt_version", getattr(args, "llm_prompt_version", "default")])
    return cmd


def run_metric_group_subprocess(
    args: argparse.Namespace,
    group: str,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run one metric group in a clean subprocess and load its partial outputs."""
    os.makedirs(args.output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"phase3_{group}_", dir=args.output_dir) as tmp_dir:
        partial_results_path = os.path.join(tmp_dir, f"{group}_results.json")
        partial_corpus_path = os.path.join(tmp_dir, f"{group}_corpus.json")
        cmd = build_subprocess_command(args, group, partial_results_path, partial_corpus_path)
        print(f"[Phase 3] Running isolated metric group: {group}")
        subprocess.run(cmd, check=True)
        return load_json(partial_results_path), load_json(partial_corpus_path)


def load_precomputed_group_results(
    args: argparse.Namespace,
    group_results_dir: str,
    records: List[EvalRecord],
) -> tuple[List[List[Dict[str, Any]]], Dict[str, float]]:
    """Load previously computed group outputs for final aggregation."""
    all_results: List[List[Dict[str, Any]]] = []
    corpus_bleu: Dict[str, float] = {}

    for group in get_phase3_groups(args, allow_llm_without_url=True):
        group_results_path = os.path.join(group_results_dir, f"{group}_results.json")
        shard_results_dir = os.path.join(group_results_dir, group)
        if not os.path.exists(group_results_path) and not os.path.isdir(shard_results_dir):
            continue

        sparse_rows = load_sparse_group_rows(
            group_results_path=group_results_path,
            shard_results_dir=shard_results_dir,
        )
        all_results.append(align_sparse_rows_to_records(records, sparse_rows))

        group_corpus_path = os.path.join(group_results_dir, f"{group}_corpus.json")
        if os.path.exists(group_corpus_path):
            corpus_bleu.update(load_json(group_corpus_path))

    return all_results, corpus_bleu


def build_metric_defaults(args: argparse.Namespace) -> Dict[str, Any]:
    """Return fail-safe fallback values for metrics enabled in this run."""
    asr_basic_only = getattr(args, "asr_basic_only", False)
    defaults: Dict[str, Any] = {"bleu_asr": 0.0}

    if not asr_basic_only:
        defaults["bleu"] = 0.0
        defaults["duration_ratio"] = 0.0
        thresholds = [float(t) for t in args.slc_thresholds.split(",") if t]
        for threshold in thresholds:
            defaults[f"slc_{threshold}"] = False

    if args.enable_comet and args.base_comet_model:
        defaults["comet_ref_asr"] = 0.0
        if not asr_basic_only:
            defaults["comet_ref"] = 0.0
        if args.comet_qe_model:
            defaults["comet_qe_asr"] = 0.0
            if not asr_basic_only:
                defaults["comet_qe"] = 0.0

    if args.enable_xcomet and args.xcomet_model:
        defaults["xcomet_qe_asr"] = 0.0
        defaults["xcomet_ref_asr"] = 0.0
        if not asr_basic_only:
            defaults["xcomet_qe"] = 0.0
            defaults["xcomet_ref"] = 0.0

    if args.enable_speaker_sim:
        defaults["speaker_similarity"] = 0.0

    if args.enable_llm:
        if not getattr(args, "disable_llm_emotion", False):
            defaults["emotion_score"] = 1.0
        if not getattr(args, "disable_llm_style", False):
            defaults["style_score"] = 1.0
        if not getattr(args, "disable_llm_event", False):
            defaults["event_score"] = 1.0

    return defaults


def apply_metric_defaults(
    rows: List[Dict[str, Any]],
    metric_defaults: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Fill missing enabled metrics with run-policy fallback values."""
    for row in rows:
        for key, value in metric_defaults.items():
            row.setdefault(key, value)
    return rows


def compute_summary(merged: List[Dict[str, Any]], corpus_bleu: Dict[str, float]) -> Dict[str, Any]:
    """Compute corpus-level summary statistics."""
    if not merged:
        return {}

    summary: Dict[str, Any] = {
        "model_name": merged[0].get("model_name", "unknown"),
        "src_lang": merged[0].get("src_lang", ""),
        "tgt_lang": merged[0].get("tgt_lang", ""),
        "n_total": len(merged),
        "n_audio": sum(1 for r in merged if r.get("has_audio")),
        "n_text_only": sum(1 for r in merged if not r.get("has_audio")),
        "n_errors": sum(1 for r in merged if r.get("error") is not None),
    }

    # Corpus-level BLEU
    summary.update(corpus_bleu)

    # Mean metrics
    mean_keys = [
        "bleu", "bleu_asr",
        "comet_qe", "comet_ref", "comet_qe_asr", "comet_ref_asr",
        "xcomet_qe", "xcomet_ref", "xcomet_qe_asr", "xcomet_ref_asr",
        "duration_ratio", "speaker_similarity",
        "emotion_score", "style_score", "event_score",
    ]
    for key in mean_keys:
        vals = [r[key] for r in merged if isinstance(r.get(key), (int, float))]
        if vals:
            summary[f"{key}_mean"] = sum(vals) / len(vals)

    # SLC pass rates (dynamic keys — scan all records in case first has no SLC data)
    slc_keys: set = set()
    for row in merged:
        slc_keys.update(k for k in row if k.startswith("slc_"))
    for key in slc_keys:
        vals = [1 if r.get(key) else 0 for r in merged if key in r]
        if vals:
            summary[f"{key}_pass_rate"] = sum(vals) / len(vals)

    return summary


# ---- Main ----

def run_pipeline(args: argparse.Namespace) -> None:
    records = load_eval_records(args.input)
    if not records:
        print("No records loaded; exit.")
        return

    print(f"Loaded {len(records)} eval records")
    aggregate_group_results_dir = getattr(args, "aggregate_group_results_dir", None)
    if aggregate_group_results_dir:
        all_results, corpus_bleu = load_precomputed_group_results(
            args, aggregate_group_results_dir, records
        )
    else:
        all_results = []
        corpus_bleu = {}
        for group in get_phase3_groups(args):
            if getattr(args, "isolate_metric_groups", False):
                group_rows, group_corpus_bleu = run_metric_group_subprocess(args, group)
            else:
                group_rows, group_corpus_bleu = run_metric_group_locally(records, args, group)
            all_results.append(group_rows)
            corpus_bleu.update(group_corpus_bleu)

    # --- Phase 6: Aggregation ---
    print("[Phase 6] Merging results...")
    merged = merge_scorer_results(records, all_results)
    merged = apply_metric_defaults(merged, build_metric_defaults(args))
    summary = compute_summary(merged, corpus_bleu)

    # Determine model name for output filenames
    model_name = records[0].model_name if records else "unknown"
    # Sanitize for filename
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")

    results_path = os.path.join(args.output_dir, f"eval_results_{safe_model_name}.jsonl")
    summary_path = os.path.join(args.output_dir, f"eval_summary_{safe_model_name}.json")

    write_jsonl(results_path, merged)
    print(f"Saved results to {results_path}")

    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation orchestrator")
    p.add_argument("--input", required=True, help="Input eval_records JSONL")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--src_lang", default="zh")
    p.add_argument("--tgt_lang", default="en")
    # COMET
    p.add_argument("--base_comet_model", default=None)
    p.add_argument("--xcomet_model", default=None)
    p.add_argument("--comet_qe_model", default=None)
    p.add_argument("--enable_comet", action="store_true")
    p.add_argument("--enable_xcomet", action="store_true")
    p.add_argument(
        "--asr_basic_only",
        action="store_true",
        help="Compute only ASR-dependent basic metrics within enabled metric groups.",
    )
    p.add_argument(
        "--disable_basic_audio",
        action="store_true",
        help="Skip the default BLEU/duration/SLC metric group.",
    )
    # Speaker sim
    p.add_argument("--enable_speaker_sim", action="store_true")
    p.add_argument(
        "--speaker_sim_ckpt",
        default=os.environ.get("SPEAKER_SIM_CKPT"),
        help="Path to wavlm_large_finetune.pth. Falls back to $SPEAKER_SIM_CKPT.",
    )
    p.add_argument(
        "--speaker_sim_python",
        default=os.environ.get("SPEAKER_SIM_PYTHON"),
        help="Isolated env python for speaker_sim workers. Falls back to $SPEAKER_SIM_PYTHON.",
    )
    # LLM
    p.add_argument("--enable_llm", action="store_true")
    p.add_argument("--disable_llm_emotion", action="store_true")
    p.add_argument("--disable_llm_style", action="store_true")
    p.add_argument("--disable_llm_event", action="store_true")
    p.add_argument("--llm_url", default=None)
    p.add_argument("--llm_model", default=None)
    p.add_argument("--llm_concurrency", type=int, default=100)
    p.add_argument("--llm_ensemble_runs", type=int, default=3)
    p.add_argument(
        "--llm_ensemble_strategy",
        default="robust",
        help="Aggregation strategy for repeated LLM judging: robust, median, mean, or majority.",
    )
    p.add_argument("--llm_prompt_version", default="default")
    # SLC
    p.add_argument("--slc_thresholds", default="0.2,0.4")
    p.add_argument("--isolate_metric_groups", action="store_true")
    # Internal subprocess split controls
    p.add_argument("--score_group", default=None)
    p.add_argument("--partial_results_path", default=None)
    p.add_argument("--partial_corpus_path", default=None)
    p.add_argument("--aggregate_group_results_dir", default=None)
    p.add_argument("--phase3_rank", type=int, default=0)
    p.add_argument("--phase3_world_size", type=int, default=1)
    p.add_argument("--existing_group_results_path", default=None)
    p.add_argument("--shard_results_dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_group:
        records = load_eval_records(args.input)
        if args.phase3_world_size > 1:
            required_keys = None
            if args.score_group == "speaker_sim":
                required_keys = {"speaker_similarity"}
            existing_rows = load_sparse_group_rows(
                group_results_path=args.existing_group_results_path,
                shard_results_dir=args.shard_results_dir,
            )
            completed_ids = collect_completed_ids_from_rows(
                existing_rows,
                required_keys=required_keys,
            )
            records = build_pending_shard_records(
                records,
                completed_ids=completed_ids,
                rank=args.phase3_rank,
                world_size=args.phase3_world_size,
            )
        rows, corpus_bleu = run_metric_group_locally(records, args, args.score_group)
        if not args.partial_results_path or not args.partial_corpus_path:
            raise ValueError("partial output paths are required when score_group is set")
        write_json(args.partial_results_path, rows)
        write_json(args.partial_corpus_path, corpus_bleu)
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
