#!/usr/bin/env python3
"""
Qwen3-ASR + ForcedAligner: ASR transcription with word-level timestamps.

Uses Qwen3ASRModel.LLM(forced_aligner=...) to simultaneously produce ASR text and
word-level timestamps in a single pass.

Multi-GPU support: Each GPU runs an independent process (launched via shell with
CUDA_VISIBLE_DEVICES). File-level coordination is handled by file_task_manager
(atomic directory locks) to avoid duplicate processing.

Output:
  Standard JSONL  — one file per parquet, mirroring input structure (for Event Combine)
  Meta JSONL      — merged 0.jsonl per leaf directory for downstream metadata consumers
"""

import os
import json
import torch
import sys
import argparse
import io
import soundfile as sf
from collections import Counter, defaultdict
from distutils.util import strtobool
from typing import Dict, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import parquet_io
from utils.file_task_manager import discover_tasks, process_all_tasks, get_task_status
from utils.qwen_asr_vllm_compat import apply_patch as apply_qwen_asr_vllm_compat

CHUNK_SIZE = 20000

global_model = None


# ---------------------------------------------------------------------------
# Helper functions (shared with process_qwen3.py)
# ---------------------------------------------------------------------------

def get_column_value(row: Dict, possible_columns: List[str]):
    for col in possible_columns:
        if col in row and row[col] is not None:
            return row[col]
    return None


def resolve_lang(tts_lang):
    if tts_lang is None:
        return None
    if hasattr(tts_lang, "__len__") and not isinstance(tts_lang, str):
        if hasattr(tts_lang, "tolist"):
            l_list = tts_lang.tolist()
        else:
            l_list = list(tts_lang)
        l_list = [str(x) for x in l_list if x]
        if not l_list:
            return None
        common = Counter(l_list).most_common(1)
        return common[0][0] if common else None
    return str(tts_lang)


def infer_lang_from_path(input_path: str):
    path_lower = input_path.lower().replace("\\", "/")
    if "/chinese/" in path_lower or "/zh/" in path_lower:
        return "zh"
    elif "/english/" in path_lower or "/en/" in path_lower:
        return "en"
    return None


def get_language_for_aligner(row: Dict, input_path: str) -> str:
    """Return language string accepted by Qwen3ASRModel ('Chinese' / 'English')."""
    lang_cols = ["lang", "language", "lid", "original_language"]
    current_val = get_column_value(row, lang_cols)
    resolved = resolve_lang(current_val)
    if not resolved:
        resolved = infer_lang_from_path(input_path)
    if resolved:
        lang = str(resolved).lower().strip()
        if lang in ["zh", "cn", "chinese", "中文"]:
            return "Chinese"
        elif lang in ["en", "english", "英文"]:
            return "English"
    return "Chinese"


def decode_audio_bytes(audio_bytes):
    """Decode raw audio bytes → (np.ndarray float32, sample_rate). No resampling."""
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        return None, None
    if audio is None:
        return None, None
    if hasattr(audio, "ndim") and audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, int(sr)


# ---------------------------------------------------------------------------
# Single-file processing (used by file_task_manager)
# ---------------------------------------------------------------------------

def process_chunk(pq_file: str, output_jsonl_path: str, batch_size: int):
    """Process a single .base.parquet file: ASR + timestamps → JSONL."""
    if global_model is None:
        print(f"[ERROR] Model not initialized, skipping file")
        return

    # --- Resume support: load already-processed IDs ---
    processed_ids = set()
    if os.path.exists(output_jsonl_path):
        try:
            with open(output_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "id" in record:
                            processed_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue
            print(
                f"Found {len(processed_ids)} already processed IDs "
                f"in {os.path.basename(output_jsonl_path)}"
            )
        except Exception as e:
            print(f"[WARN] Failed to read existing output file: {e}")

    # --- Read parquet ---
    try:
        items = parquet_io.read_parquet_safe(pq_file)
    except Exception as e:
        print(f"[ERROR] Failed to read parquet {pq_file}: {e}")
        return

    if not items:
        print(f"Empty parquet: {pq_file}")
        return

    # --- Collect valid samples ---
    valid_items = []
    valid_audios = []
    valid_languages = []

    for entry in items:
        sample_id = entry.get("id", "unknown")
        if sample_id in processed_ids:
            continue

        audio_data = entry.get("audio")
        if not audio_data:
            continue

        audio_bytes = None
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_bytes = audio_data["bytes"]
        elif isinstance(audio_data, (bytes, bytearray)):
            audio_bytes = bytes(audio_data)

        if not audio_bytes:
            continue

        audio, sr = decode_audio_bytes(audio_bytes)
        if audio is None:
            continue

        language = get_language_for_aligner(entry, pq_file)

        valid_items.append({"id": sample_id})
        valid_audios.append((audio, sr))  # pass original sr — model handles resampling
        valid_languages.append(language)

    total_samples = len(items)
    to_process = len(valid_items)
    already_done = len(processed_ids)

    print(
        f"{os.path.basename(pq_file)}: "
        f"Total={total_samples}, Done={already_done}, ToProcess={to_process}"
    )

    if to_process == 0:
        print(f"Nothing to process.")
        return

    # --- Batch inference ---
    results = []
    failed_batches = 0
    for batch_start in range(0, to_process, batch_size):
        batch_end = min(batch_start + batch_size, to_process)
        b_items = valid_items[batch_start:batch_end]
        b_audios = valid_audios[batch_start:batch_end]
        b_langs = valid_languages[batch_start:batch_end]

        try:
            asr_results = global_model.transcribe(
                audio=b_audios,
                language=b_langs,
                return_time_stamps=True,
            )

            for item_info, asr_r in zip(b_items, asr_results):
                text = asr_r.text if hasattr(asr_r, "text") else str(asr_r)
                detected_lang = getattr(asr_r, "language", None)

                triples = []
                if hasattr(asr_r, "time_stamps") and asr_r.time_stamps:
                    for ts in asr_r.time_stamps:
                        triples.append([
                            ts.text,
                            int(ts.start_time * 1000),
                            int(ts.end_time * 1000),
                        ])

                results.append({
                    "id": item_info["id"],
                    "text": text,
                    "punctuated_text": text,
                    "timestamp": triples,
                    "language": detected_lang,
                })
        except Exception as e:
            failed_batches += 1
            print(
                f"[ERROR] Failed batch {batch_start}-{batch_end}: {e}"
            )
            import traceback
            traceback.print_exc()
            continue

    # --- Write results (append for resume) ---
    if results:
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"Wrote {len(results)} results "
            f"to {os.path.basename(output_jsonl_path)}"
        )
    else:
        print(f"No results to write.")

    if failed_batches:
        raise RuntimeError(
            f"{failed_batches} ASR batch(es) failed while processing {pq_file}; "
            "leaving task unfinished for resume."
        )


# ---------------------------------------------------------------------------
# Meta output generation
# ---------------------------------------------------------------------------

def generate_meta_output(output_dir: str, meta_output_dir: str, input_dir: str):
    """
    Generate meta-compatible output for downstream metadata consumers.

    Historical metadata consumers resolved JSONL paths as:
      1. Strip '_parquets' suffix from path components
      2. Always looks for '0.jsonl'

    So for each leaf directory under output_dir we:
      - Strip '_parquets' from path components
      - Merge all JSONL files into a single 0.jsonl
    """
    print(f"\n[Meta] Generating meta output: {meta_output_dir}")

    # Collect all JSONL files grouped by directory
    dir_files = defaultdict(list)
    for root, _dirs, files in os.walk(output_dir):
        for fname in sorted(files):
            if fname.endswith(".jsonl"):
                dir_files[root].append(os.path.join(root, fname))

    total_files = sum(len(v) for v in dir_files.values())
    total_records = 0

    for src_dir, jsonl_files in dir_files.items():
        # Compute relative path from output_dir
        rel_path = os.path.relpath(src_dir, output_dir)

        # Strip _parquets suffix from path components (mimic get_meta_jsonl_path)
        parts = rel_path.split(os.sep)
        new_parts = []
        for p in parts:
            if p.endswith("_parquets"):
                new_parts.append(p[:-9])
            else:
                new_parts.append(p)
        new_rel_path = os.path.join(*new_parts) if new_parts else ""

        target_dir = os.path.join(meta_output_dir, new_rel_path)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "0.jsonl")

        # Merge all JSONL files in this directory into one 0.jsonl
        # Use a set to deduplicate by id
        seen_ids = set()
        records = []

        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        sid = item.get("id")
                        if sid and sid not in seen_ids:
                            seen_ids.add(sid)
                            records.append(item)
                    except json.JSONDecodeError:
                        continue

        if records:
            with open(target_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_records += len(records)

    print(
        f"[Meta] Done: {total_files} source files → {len(dir_files)} directories, "
        f"{total_records} total records"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR + ForcedAligner: single-GPU ASR with word-level timestamps. "
                    "For multi-GPU, launch multiple processes via shell with CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Input directory containing .base.parquet files",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for standard JSONL results (for Event Combine)",
    )
    parser.add_argument(
        "--meta_output_dir", default=None,
        help="Meta-compatible output directory (for translate --meta_root). "
             "Default: <output_dir>_meta",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="(Deprecated, ignored) Number of GPUs — now controlled by shell CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--threads_per_gpu", type=int, default=1,
        help="(Deprecated, ignored) Worker processes per GPU",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--asr_model_path",
        default=os.environ.get("ASR_MODEL_PATH", ""),
        help="Path to Qwen3-ASR model",
    )
    parser.add_argument(
        "--aligner_model_path",
        default=os.environ.get("ALIGNER_MODEL_PATH", ""),
        help="Path to Qwen3-ForcedAligner model",
    )
    parser.add_argument(
        "--file_suffix", default=".base.parquet",
        help="Parquet file suffix to match (default: .base.parquet)",
    )
    parser.add_argument(
        "--skip_meta", action="store_true",
        help="Skip meta output generation (useful when running multiple GPU processes — "
             "generate meta only once after all processes finish)",
    )
    parser.add_argument(
        "--enforce_eager",
        default=os.environ.get("ASR_ENFORCE_EAGER", "0"),
        help="Whether to force vLLM eager mode for Qwen3-ASR. Accepts 1/0 or true/false.",
    )

    args = parser.parse_args()

    if not args.asr_model_path or not args.aligner_model_path:
        parser.error(
            "--asr_model_path and --aligner_model_path are required "
            "(or set ASR_MODEL_PATH and ALIGNER_MODEL_PATH)."
        )

    if args.meta_output_dir is None:
        args.meta_output_dir = args.output_dir + "_meta"

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model on current GPU (CUDA_VISIBLE_DEVICES set by shell) ---
    global global_model
    gpu_info = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    print(f"Loading Qwen3ASRModel.LLM (CUDA_VISIBLE_DEVICES={gpu_info})...")

    # Set VLLM_TARGET_DEVICE before importing vllm (via qwen_asr) — required on
    # AMD ROCm where the .qwen3-asr venv's vllm cannot auto-detect the platform.
    if "VLLM_TARGET_DEVICE" not in os.environ:
        if hasattr(torch.version, "hip") and torch.version.hip:
            os.environ["VLLM_TARGET_DEVICE"] = "rocm"
            print(f"Auto-detected ROCm (HIP {torch.version.hip}), set VLLM_TARGET_DEVICE=rocm")

    try:
        apply_qwen_asr_vllm_compat()
        from qwen_asr import Qwen3ASRModel

        enforce_eager = bool(strtobool(str(args.enforce_eager)))
        global_model = Qwen3ASRModel.LLM(
            model=args.asr_model_path,
            gpu_memory_utilization=0.8,
            max_inference_batch_size=32,
            max_new_tokens=1024,
            enforce_eager=enforce_eager,
            forced_aligner=args.aligner_model_path,
            forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
        )
        print(f"Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # --- Discover tasks via file_task_manager ---
    tasks = discover_tasks(
        args.input_dir,
        args.output_dir,
        suffix=args.file_suffix,
        output_suffix=".jsonl",
    )
    print(f"Discovered {len(tasks)} task files")

    if not tasks:
        print("No files to process.")
        return

    status = get_task_status(tasks)
    print(f"Task status: {status}")

    # --- Process files with file-level locking ---
    batch_size = args.batch_size

    def process_one_file(task):
        process_chunk(task["input_path"], task["output_path"], batch_size)

    stats = process_all_tasks(tasks, process_one_file)
    print(f"Processing complete: {stats}")
    if stats.get("errors", 0):
        print(f"[ERROR] {stats['errors']} task(s) failed; exiting non-zero for retry.")
        return 1

    # --- Generate meta-compatible output (skip if --skip_meta) ---
    if not args.skip_meta:
        generate_meta_output(args.output_dir, args.meta_output_dir, args.input_dir)

    print("All done.")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
