"""Step 6: Merge extracted hypothesis features back into the eval JSONL.

Reads intermediate JSONL outputs from ASR/Caption/Summary/Event stages,
joins by sample id, and writes a single JSONL with all hyp_* fields populated.

Usage:
    python merge_hyp_features.py \
        --input_jsonl eval_input.jsonl \
        --asr_dir /tmp/eval_work/hyp_timestamp \
        --caption_dir /tmp/eval_work/hyp_caption \
        --summary_dir /tmp/eval_work/hyp_summary \
        --event_dir /tmp/eval_work/hyp_events \
        --output_jsonl /tmp/eval_work/eval_merged.jsonl
"""
import argparse
import json
import os
import glob


def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_dir_jsonl(dir_path):
    """Load all JSONL files from a directory, return dict keyed by id."""
    result = {}
    if not dir_path or not os.path.exists(dir_path):
        return result

    if os.path.isfile(dir_path):
        files = [dir_path]
    else:
        files = sorted(glob.glob(os.path.join(dir_path, '**', '*.jsonl'), recursive=True))

    for f in files:
        for rec in load_jsonl(f):
            rid = rec.get('id')
            if rid:
                result[rid] = rec
    return result


def main():
    parser = argparse.ArgumentParser(description='Merge hypothesis features into eval JSONL.')
    parser.add_argument('--input_jsonl', required=True, help='Original eval input JSONL')
    parser.add_argument('--asr_dir', default=None, help='Directory with ASR timestamp JSONL outputs')
    parser.add_argument('--caption_dir', default=None, help='Directory with caption JSONL outputs')
    parser.add_argument('--summary_dir', default=None, help='Directory with emotion/style summary JSONL outputs')
    parser.add_argument('--event_dir', default=None, help='Directory with event combine JSONL outputs')
    parser.add_argument('--output_jsonl', required=True, help='Output merged JSONL path')
    args = parser.parse_args()

    # Load original records
    records = load_jsonl(args.input_jsonl)
    if not records:
        print('[ERROR] No records in input JSONL.')
        return

    # Load intermediate results
    asr_map = load_dir_jsonl(args.asr_dir)
    caption_map = load_dir_jsonl(args.caption_dir)
    summary_map = load_dir_jsonl(args.summary_dir)
    event_map = load_dir_jsonl(args.event_dir)

    print(f'Loaded intermediates — ASR: {len(asr_map)}, Caption: {len(caption_map)}, '
          f'Summary: {len(summary_map)}, Events: {len(event_map)}')

    # Merge
    for rec in records:
        rid = rec.get('id')
        if not rid:
            continue

        # ASR → hyp_asr_text (transcription text) + hyp_timestamp
        asr_rec = asr_map.get(rid, {})
        if asr_rec:
            rec['hyp_asr_text'] = asr_rec.get('text') or asr_rec.get('punctuated_text', '')
            rec['hyp_timestamp'] = asr_rec.get('timestamp_prediction', '')

        # Caption → hyp_caption
        cap_rec = caption_map.get(rid, {})
        if cap_rec:
            rec['hyp_caption'] = cap_rec.get('caption', '')

        # Summary → hyp_emotion, hyp_style
        sum_rec = summary_map.get(rid, {})
        if sum_rec:
            rec['hyp_emotion'] = sum_rec.get('emotion', '')
            rec['hyp_style'] = sum_rec.get('style', '')

        # Events → hyp_asr_text_with_event
        evt_rec = event_map.get(rid, {})
        if evt_rec:
            text_with_events = evt_rec.get('text_with_events', '')
            rec['hyp_asr_text_with_event'] = text_with_events
            # Backward-compatible alias for older analysis tools.
            rec['hyp_text_with_events'] = text_with_events

    # Write
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'Wrote {len(records)} merged records to {args.output_jsonl}')


if __name__ == '__main__':
    main()
