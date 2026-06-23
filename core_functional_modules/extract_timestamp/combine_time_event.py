"""
将timestamp与event结合起来，将text中混合音频事件
"""

import os
import re
import json
import argparse
from typing import List, Dict, Tuple
import glob
import pandas as pd

# Default Event Map
# Modify this dictionary to control which events are inserted and their tags
EVENT_LABEL_TO_TAG = {
    "Breathing": "[Breathing]",
    "Cough": "[Cough]",
    "Laughter": "[Laughter]",
    "Sneeze": "[Sneeze]",
    "Crying, sobbing": "[Crying]",
    "Whispering": "[Whispering]",
    "Sigh": "[Sigh]",
    "Pant": "[Pant]",
    "Burping, eructation": "[Burp]",
    # "Background noise": "<noise>", # Usually output as a span, might clutter text
    # "Speech synthesizer": "<synth>",
}

EVENT_TAG_ALIASES = {
    "breathing": "[Breathing]",
    "breath": "[Breathing]",
    "inhale": "[Breathing]",
    "exhale": "[Breathing]",
    "heavy breath": "[Breathing]",
    "short pause": "[Breathing]",
    "pause": "[Breathing]",
    "slight pause": "[Breathing]",
    "long pause": "[Breathing]",
    "paused": "[Breathing]",
    "laughter": "[Laughter]",
    "laughing": "[Laughter]",
    "laughs": "[Laughter]",
    "laugh": "[Laughter]",
    "light laugh": "[Laughter]",
    "light laughter": "[Laughter]",
    "chuckle": "[Laughter]",
    "chuckling": "[Laughter]",
    "giggle": "[Laughter]",
    "cough": "[Cough]",
    "coughing": "[Cough]",
    "throat clear": "[Cough]",
    "clearing throat": "[Cough]",
    "sigh": "[Sigh]",
    "sighing": "[Sigh]",
    "sigh of relief": "[Sigh]",
    "resigned sigh": "[Sigh]",
    "slightly sighing": "[Sigh]",
    "pant": "[Pant]",
    "panting": "[Pant]",
    "whispering": "[Whispering]",
    "whisper": "[Whispering]",
    "whisper in small voice": "[Whispering]",
    "crying": "[Crying]",
}


def normalize_event_tag(label: str) -> str | None:
    if not isinstance(label, str):
        return None
    if label in EVENT_LABEL_TO_TAG:
        return EVENT_LABEL_TO_TAG[label]
    normalized = re.sub(r"\s+", " ", label.strip().lower())
    return EVENT_TAG_ALIASES.get(normalized)


def _is_punctuation_only(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return re.search(r"\w", stripped) is None


_ASCII_TOKEN_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _compile_ascii_token_pattern(token: str) -> re.Pattern:
    if token in _ASCII_TOKEN_PATTERN_CACHE:
        return _ASCII_TOKEN_PATTERN_CACHE[token]

    flags = re.IGNORECASE if re.search(r"[A-Za-z]", token) else 0

    if token and token.isascii() and re.fullmatch(r"[A-Za-z]+", token):
        if len(token) == 1:
            core = re.escape(token)
        else:
            parts = []
            for ch in token[:-1]:
                parts.append(re.escape(ch))
                parts.append(r"(?:['\u2019])?")
            parts.append(re.escape(token[-1]))
            core = "".join(parts)
    else:
        core = re.escape(token)

    if re.search(r"[A-Za-z0-9]", token):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])", flags)
    else:
        pattern = re.compile(core, flags)

    _ASCII_TOKEN_PATTERN_CACHE[token] = pattern
    return pattern


def _find_token_span(raw_text: str, token: str, cursor: int) -> Tuple[int, int]:
    if not token:
        return (-1, -1)
    if token.isascii():
        m = _compile_ascii_token_pattern(token).search(raw_text, cursor)
        return (m.start(), m.end()) if m else (-1, -1)

    idx = raw_text.find(token, cursor)
    return (idx, idx + len(token)) if idx != -1 else (-1, -1)


def load_events(event_path: str, threshold: str = "0.5") -> Dict[str, List[Dict]]:
    """
    Load events from a JSONL file or directory of JSONL files.
    Returns a dictionary mapping file_id to a list of events.
    """
    events_map = {}

    if os.path.isdir(event_path):
        files = glob.glob(os.path.join(event_path, "**/*.jsonl"), recursive=True)
        # Also try just .json if strictly sed output
        if not files:
            files = glob.glob(os.path.join(event_path, "**/*.json"), recursive=True)
    else:
        files = [event_path]

    print(f"Loading events from {len(files)} files in {event_path}...")

    for file_p in files:
        with open(file_p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    file_id = data.get("id")
                    # Also try to handle if id is different or needs mapping

                    # The event structure: {"id": "...", "events": {"0.5": [...], ...}}
                    all_events = data.get("events", {})

                    # Check if 'events' is a dict (multi-threshold) or list (single)
                    if isinstance(all_events, dict):
                        if threshold in all_events:
                            events_list = all_events[threshold]
                        else:
                            # Fallback to the first available key if threshold not found
                            if all_events:
                                first_key = list(all_events.keys())[0]
                                events_list = all_events[first_key]
                            else:
                                events_list = []
                    elif isinstance(all_events, list):
                        events_list = all_events
                    else:
                        events_list = []

                    if file_id:
                        events_map[file_id] = events_list

                        # Check first event for filename
                        if events_list and "filename" in events_list[0]:
                            filename = events_list[0]["filename"]
                            basename = os.path.splitext(os.path.basename(filename))[0]
                            if basename != file_id:
                                events_map[basename] = events_list

                except json.JSONDecodeError:
                    continue
    return events_map


def merge_text_and_events(item: Dict, events: List[Dict]) -> str:
    """Merge text with events by inserting event tags after corresponding words.

    Strategy: Keep raw_text order intact, only insert event tags at appropriate positions.

    item: {"text": "Raw text...", "timestamp": [["word", start, end], ...]}
    events: List of {"event_label": "...", "onset": 0.0, "offset": 1.0, ...}
    """
    raw_text = item.get("text", "")
    timestamps = item.get("timestamp", [])

    if not raw_text or not timestamps:
        return raw_text

    valid_events = []
    for event in events:
        label = event.get("event_label")
        tag = normalize_event_tag(label)
        if tag:
            valid_events.append(
                {
                    "tag": tag,
                    "onset": float(event.get("onset", 0.0)),
                }
            )

    if not valid_events:
        return raw_text

    word_spans = []
    cursor = 0

    for word, start_ms, end_ms in timestamps:
        start = start_ms / 1000.0
        end = end_ms / 1000.0

        match_start, match_end = _find_token_span(raw_text, word, cursor)
        if match_start == -1:
            continue

        word_spans.append(
            {
                "text_start": match_start,
                "text_end": match_end,
                "time_start": start,
                "time_end": end,
            }
        )
        cursor = match_end

    if not word_spans:
        return raw_text

    events_to_insert = {}

    for event in valid_events:
        onset = event["onset"]
        tag = event["tag"]

        insert_after_idx = -1
        for i, span in enumerate(word_spans):
            if span["time_end"] <= onset:
                insert_after_idx = i
            elif span["time_start"] <= onset < span["time_end"]:
                insert_after_idx = i
                break
            else:
                break

        if insert_after_idx >= 0:
            text_pos = word_spans[insert_after_idx]["text_end"]
        else:
            text_pos = word_spans[0]["text_start"]

        events_to_insert.setdefault(text_pos, []).append(tag)

    result = []
    last_pos = 0
    for pos in sorted(events_to_insert.keys()):
        result.append(raw_text[last_pos:pos])
        result.extend(events_to_insert[pos])
        last_pos = pos

    result.append(raw_text[last_pos:])

    return "".join(result)


def load_meta_text_map(meta_root):
    """Load text map from meta files if available"""
    meta_map = {}
    if not meta_root:
        return meta_map

    files = []
    if os.path.isdir(meta_root):
        files = glob.glob(os.path.join(meta_root, "**/*.jsonl"), recursive=True)
    elif os.path.exists(meta_root):
        files = [meta_root]

    print(f"Loading meta from {len(files)} files...")
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if "id" in item and "punctuated_text" in item:
                                meta_map[str(item["id"])] = item["punctuated_text"]
                        except:
                            pass
        except Exception:
            pass
    return meta_map


def load_tts_text_map(parquet_dir):
    """Load text from tts parquet files"""
    tts_map = {}
    if not parquet_dir or not os.path.exists(parquet_dir):
        return tts_map

    files = glob.glob(os.path.join(parquet_dir, "**/*.tts.parquet"), recursive=True)
    if not files:
        # Fallback to recursively finding in subdirs is handled by glob
        pass

    print(f"Loading TTS text from {len(files)} files...")
    for fpath in files:
        try:
            df = None
            # Try to read with punctuated_text first
            try:
                df = pd.read_parquet(fpath, columns=["id", "text", "punctuated_text"])
            except Exception:
                # Fallback: maybe punctuated_text does not exist
                try:
                    df = pd.read_parquet(fpath, columns=["id", "text"])
                except Exception:
                    pass

            if df is not None:
                # Replace nan with None to handle missing values safely
                df = df.where(pd.notnull(df), None)
                records = df.to_dict("records")
                for r in records:
                    tts_map[str(r["id"])] = r
        except Exception:
            pass
    return tts_map


def main():
    parser = argparse.ArgumentParser(
        description="Merge timestamped text with audio events."
    )
    parser.add_argument(
        "--timestamp_file",
        type=str,
        required=True,
        help="Path to input JSONL file with timestamps.",
    )
    parser.add_argument(
        "--event_file",
        type=str,
        required=True,
        help="Path to input JSONL file with SED events.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file."
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default=None,
        help="Root directory for original parquet files (to find .tts.parquet).",
    )
    parser.add_argument(
        "--meta_root",
        type=str,
        default=None,
        help="Root directory for meta jsonl files.",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="0.5",
        help="Event confidence threshold key (e.g. 0.5).",
    )

    args = parser.parse_args()

    if os.path.isdir(args.timestamp_file):
        # Batch process if input is a directory -> Recursive
        input_files = glob.glob(
            os.path.join(args.timestamp_file, "**/*.jsonl"), recursive=True
        )
        output_is_dir = True
        if not os.path.exists(args.output_file):
            os.makedirs(args.output_file)
    else:
        input_files = [args.timestamp_file]
        output_is_dir = False
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print(f"Found {len(input_files)} timestamp files to process.")
    events_map = load_events(args.event_file, args.threshold)
    print(f"Loaded events for {len(events_map)} files.")

    # Load External Text sources
    tts_map = load_tts_text_map(args.parquet_dir)
    meta_map = load_meta_text_map(args.meta_root)

    for in_file in input_files:
        if output_is_dir:
            # Maintain subdirectory structure relative to args.timestamp_file
            rel_path = os.path.relpath(in_file, args.timestamp_file)
            out_file = os.path.join(args.output_file, rel_path)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        else:
            out_file = args.output_file

        print(f"Processing {in_file} -> {out_file}...")

        with (
            open(in_file, "r", encoding="utf-8") as fin,
            open(out_file, "w", encoding="utf-8") as fout,
        ):
            processed_count = 0
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        continue
                    item_id = str(item.get("id") or item.get("index"))

                    # Correct Text Source Priority:
                    # 1. TTS Parquet (punctuated_text)
                    # 2. Meta File (punctuated_text)
                    # 3. TTS Parquet (text)
                    # 4. Item itself (text)

                    base_text = None

                    # 1. TTS punctuated_text
                    if item_id in tts_map:
                        t_entry = tts_map[item_id]
                        if t_entry.get("punctuated_text"):
                            base_text = t_entry["punctuated_text"]

                    # 2. Meta punctuated_text
                    if not base_text and item_id in meta_map:
                        base_text = meta_map[item_id]

                    # 3. TTS text
                    if not base_text and item_id in tts_map:
                        t_entry = tts_map[item_id]
                        if t_entry.get("text"):
                            base_text = t_entry["text"]

                    # 4. Fallback (Item text)
                    if not base_text:
                        base_text = item.get("text", "")

                    item["text"] = base_text  # Update base text for merging

                    # Try to find corresponding events
                    events = events_map.get(item_id, [])

                    # Merge
                    new_text = merge_text_and_events(item, events)

                    item["text_with_events"] = new_text

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    processed_count += 1

                except json.JSONDecodeError:
                    continue

            print(f"Finished {in_file}. {processed_count} lines processed.")


if __name__ == "__main__":
    main()
