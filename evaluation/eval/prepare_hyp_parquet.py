"""Step 0: Convert hypothesis WAV file paths to a Parquet file with audio bytes.

Pipeline modules (Caption, SED, ASR) expect Parquet input with an 'audio' column
containing {'bytes': <raw_bytes>}. This script reads WAV paths from the eval input
JSONL and packs them into that format.

Usage:
    python prepare_hyp_parquet.py \
        --input_jsonl eval_input.jsonl \
        --output_dir /tmp/eval_work/hyp_parquet
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'core_functional_modules', 'utils'))

from parquet_io import write_parquet_safe


def main():
    parser = argparse.ArgumentParser(description='Pack hypothesis WAV files into Parquet for pipeline modules.')
    parser.add_argument('--input_jsonl', required=True, help='Input evaluation JSONL (must have id + hyp_wav_path or hyp_audio fields)')
    parser.add_argument('--output_dir', required=True, help='Output directory for Parquet file(s)')
    parser.add_argument('--batch_size', type=int, default=500, help='Records per Parquet file')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print('[ERROR] No records loaded from input JSONL.')
        return

    parquet_records = []
    skipped = 0
    for rec in records:
        audio_path = rec.get('hyp_audio') or rec.get('hyp_wav_path')
        sample_id = rec.get('id')
        if not audio_path or not os.path.exists(audio_path):
            print(f'[WARN] Missing or invalid hypothesis audio path for {sample_id}: {audio_path}')
            skipped += 1
            continue
        with open(audio_path, 'rb') as af:
            audio_bytes = af.read()
        parquet_records.append({
            'id': sample_id,
            'audio': {'bytes': audio_bytes},
        })

    if not parquet_records:
        print('[ERROR] No valid audio files found.')
        return

    # Write in batches to keep file sizes manageable
    for i in range(0, len(parquet_records), args.batch_size):
        batch = parquet_records[i:i + args.batch_size]
        batch_idx = i // args.batch_size
        out_path = os.path.join(args.output_dir, f'hyp_batch_{batch_idx:04d}.base.parquet')
        write_parquet_safe(batch, out_path)
        print(f'Wrote {len(batch)} records to {out_path}')

    print(f'Done. Total: {len(parquet_records)} packed, {skipped} skipped.')


if __name__ == '__main__':
    main()
