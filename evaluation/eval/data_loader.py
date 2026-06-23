"""Load and join benchmark + results JSONL files per formats.md spec."""

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorers.base import EvalRecord


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed JSON line in {path}")
                continue
    return records


def resolve_audio_path(path: Optional[str], base_dir: str) -> Optional[str]:
    """Resolve relative audio paths against the source JSONL directory."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def extract_cot_field(hyp_text: str, field_prefix: str) -> Optional[str]:
    """Extract a field from the model's CoT output block.

    Example field_prefix: "Translation with sound events"
    """
    if not hyp_text:
        return None
    m = re.search(rf"{re.escape(field_prefix)}:\s*(.+)", hyp_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_hyp_text(hyp_text: Optional[str]) -> Dict[str, Optional[str]]:
    """Parse the 4-line CoT block from model output.

    Expected format:
        Transcription with sound events: <Chinese text>
        Emotion: <sentence>
        Style: <sentence>
        Translation with sound events: <English text>
    """
    result: Dict[str, Optional[str]] = {
        "hyp_translation": "",
        "hyp_emotion_text": None,
        "hyp_style_text": None,
        "hyp_transcription_text": None,
    }
    if not hyp_text:
        return result

    result["hyp_transcription_text"] = extract_cot_field(
        hyp_text, "Transcription with sound events"
    )
    result["hyp_emotion_text"] = extract_cot_field(hyp_text, "Emotion")
    result["hyp_style_text"] = extract_cot_field(hyp_text, "Style")

    translation = extract_cot_field(hyp_text, "Translation with sound events")
    if translation:
        result["hyp_translation"] = translation
    else:
        # Fallback: use full hyp_text
        result["hyp_translation"] = hyp_text.strip()

    return result


def resolve_hyp_text(result_row: Dict[str, Any]) -> Optional[str]:
    """Resolve the best available text hypothesis field from a results row."""
    hyp_text = result_row.get("hyp_text")
    if hyp_text:
        return str(hyp_text)

    translation_text = result_row.get("translation_text")
    if translation_text:
        return str(translation_text)

    translation = result_row.get("translation")
    if translation:
        return str(translation)

    return None


def resolve_source_text_with_event(bench_row: Dict[str, Any]) -> str:
    """Resolve the canonical source text-with-event field for event scoring."""
    for key in (
        "src_text_with_event",
        "src_text_with_events",
        "source_text_with_event",
        "source_text_with_events",
        "text_with_events",
        "ref_text_with_events",
    ):
        value = bench_row.get(key)
        if value:
            return str(value)
    return ""


def join_records(
    benchmark: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    src_lang: str,
    tgt_lang: str,
) -> List[EvalRecord]:
    """Join benchmark and results by id, producing EvalRecord list."""
    result_by_id = {r["id"]: r for r in results if "id" in r}
    default_model_name = next(
        (str(r.get("model_name")) for r in results if r.get("model_name")),
        "unknown",
    )
    joined: List[EvalRecord] = []

    for bench in benchmark:
        rid = bench.get("id")
        if not rid:
            continue
        res = result_by_id.get(rid, {})

        # Parse translation dict from benchmark
        ref_translation = bench.get("translation", {})
        if ref_translation is None:
            ref_translation = {}
        if not isinstance(ref_translation, dict):
            ref_translation = {}

        resolved_hyp_text = resolve_hyp_text(res)
        parsed = parse_hyp_text(resolved_hyp_text)
        src_text_with_event = resolve_source_text_with_event(bench)

        record = EvalRecord(
            id=rid,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            ref_text=bench.get("text", ""),
            ref_text_with_events=src_text_with_event,
            src_text_with_event=src_text_with_event,
            ref_translation=ref_translation,
            ref_emotion=bench.get("emotion", ""),
            ref_style=bench.get("style", ""),
            ref_caption=bench.get("caption", ""),
            ref_wav_path=bench.get("wav_path"),
            hyp_text=resolved_hyp_text,
            hyp_translation=str(parsed["hyp_translation"] or ""),
            hyp_wav_path=res.get("hyp_wav_path"),
            model_name=res.get("model_name", default_model_name),
            error=res.get("error") if res else "missing_result",
            hyp_emotion_text=parsed["hyp_emotion_text"],
            hyp_style_text=parsed["hyp_style_text"],
            hyp_transcription_text=parsed["hyp_transcription_text"],
        )
        joined.append(record)

    bench_ids = {r["id"] for r in benchmark if "id" in r}
    for res in results:
        rid = res.get("id")
        if rid and rid not in bench_ids:
            print(f"[WARN] Result id={rid} not found in benchmark, skipping")

    return joined


def records_to_jsonl(records: List[EvalRecord]) -> List[Dict[str, Any]]:
    """Serialize EvalRecord list to list of dicts for JSONL output."""
    return [asdict(r) for r in records]


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Join benchmark + results JSONL for evaluation.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL")
    parser.add_argument("--results", required=True, help="Path to results JSONL")
    parser.add_argument("--src_lang", default="zh", help="Source language code")
    parser.add_argument("--tgt_lang", default="en", help="Target language code")
    parser.add_argument("--output", required=True, help="Output eval_records JSONL path")
    args = parser.parse_args()

    benchmark = load_jsonl(args.benchmark)
    results = load_jsonl(args.results)
    benchmark_dir = os.path.dirname(os.path.abspath(args.benchmark))
    results_dir = os.path.dirname(os.path.abspath(args.results))
    for row in benchmark:
        row["wav_path"] = resolve_audio_path(row.get("wav_path"), benchmark_dir)
    for row in results:
        row["hyp_wav_path"] = resolve_audio_path(row.get("hyp_wav_path"), results_dir)
    print(f"Loaded {len(benchmark)} benchmark records, {len(results)} result records")

    records = join_records(benchmark, results, args.src_lang, args.tgt_lang)
    n_audio = sum(1 for r in records if r.hyp_wav_path and r.error is None)
    n_text = sum(1 for r in records if not r.hyp_wav_path and r.error is None)
    n_err = sum(1 for r in records if r.error is not None)
    print(f"Joined: {len(records)} records ({n_audio} audio, {n_text} text-only, {n_err} errors)")

    write_jsonl(args.output, records_to_jsonl(records))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
