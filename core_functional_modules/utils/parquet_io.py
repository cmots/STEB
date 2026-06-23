import os
import pyarrow.parquet as pq
import pandas as pd
import json
from tqdm import tqdm


def find_files(root_path, suffix):
    """
    Recursively find files with a specific suffix in a directory.

    Args:
        root_path (str): The root directory to search in, or a single file path.
        suffix (str): The file suffix to look for (e.g., '.parquet', 'base.parquet').

    Returns:
        List[str]: A list of absolute file paths matching the suffix.
    """
    found_files = []
    if os.path.isfile(root_path):
        if root_path.endswith(suffix):
            return [root_path]
        return []

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(suffix):
                found_files.append(os.path.join(root, file))
    return found_files


def get_output_path(input_path, input_root, output_root, output_suffix):
    """
    Determines the output file path by mirroring the input directory structure relative to the input root.
    Handles dynamic subdirectory creation for flat structures.
    """
    # Get relative path from input_root
    if os.path.isfile(input_root):
        rel_path = os.path.basename(input_path)
    else:
        rel_path = os.path.relpath(input_path, input_root)

    parts = rel_path.split(os.sep)

    dirname = os.path.dirname(rel_path)
    filename = os.path.basename(rel_path)

    # Remove extension from filename
    base_name = os.path.splitext(filename)[0]
    if base_name.endswith(".base"):
        base_name = base_name[:-5]
    elif base_name.endswith(".tts"):
        base_name = base_name[:-4]

    final_filename = base_name + output_suffix

    dir_parts = dirname.split(os.sep) if dirname else []

    if len(dir_parts) >= 2:
        last_dir = dir_parts[-1]

        if not last_dir.endswith("_parquets"):
            src_lang = last_dir
            new_subdir = f"dataset_{src_lang}_parquets"

            new_dir_parts = dir_parts + [new_subdir]
            new_rel_dir = os.path.join(*new_dir_parts)
            return os.path.join(output_root, new_rel_dir, final_filename)

    # Default: maintain structure
    return os.path.join(output_root, dirname, final_filename)


def read_parquet_safe(input_path, columns=None):
    """
    Safely reads a parquet file, handling nested structs (like audio bytes)
    that might cause issues with standard pandas/pyarrow reading.
    Returns a list of dictionaries.
    """
    records = []
    try:
        # Try standard pandas read first (fastest)
        df = pd.read_parquet(input_path, columns=columns)
        records = df.to_dict("records")
    except Exception:
        # Fallback using safe iteration
        records = list(iter_parquet_safe(input_path, columns=columns))
    return records


def iter_parquet_safe(input_path, columns=None, use_threads=False):
    """
    Generator that safely yields records from a parquet file.
    Crucial for handling 'base.parquet' files with nested audio structs that
    crash standard multithreaded readers.
    """
    try:
        pf = pq.ParquetFile(input_path)

        arrow_names = pf.schema.to_arrow_schema().names

        if columns:
            read_cols = [c for c in columns if c in arrow_names]
        else:
            read_cols = arrow_names

        for batch in pf.iter_batches(
            batch_size=1, columns=read_cols, use_threads=use_threads
        ):
            colmap = {name: batch.column(name) for name in read_cols}
            num_rows = batch.num_rows

            for i in range(num_rows):
                try:
                    rec = {}
                    for col_name in read_cols:
                        # Special handling for 'audio' column which might be a complex struct
                        if col_name == "audio":
                            if col_name not in colmap:
                                continue
                            val = colmap[col_name][i].as_py()

                            # Sanitize audio struct if needed
                            if isinstance(val, dict):
                                audio = {}
                                # Ensure 'bytes' are captured if present
                                if "bytes" in val:
                                    audio["bytes"] = val["bytes"]

                                # Handle path/ark/offset
                                path_val = val.get("path")
                                if isinstance(path_val, dict):
                                    audio["ark"] = path_val.get("ark")
                                    audio["offset"] = path_val.get("offset")
                                elif val.get("ark"):  # Flat structure support
                                    audio["ark"] = val.get("ark")
                                    audio["offset"] = val.get("offset")

                                rec[col_name] = audio
                            else:
                                rec[col_name] = val

                        elif col_name == "bytes":
                            # If bytes is a direct column
                            cval = colmap[col_name][i]
                            try:
                                rec[col_name] = cval.as_buffer().to_pybytes()
                            except:
                                rec[col_name] = cval.as_py()
                        else:
                            # Standard columns
                            rec[col_name] = colmap[col_name][i].as_py()

                    yield rec
                except Exception as e:
                    # Log error but try to continue with next row?
                    # For data safety, skipping a bad row is often better than crashing
                    continue

    except Exception as e:
        print(f"Error iterating parquet {input_path}: {e}")
        # If even ParquetFile fails, try pandas as last resort if it wasn't tried?
        # But usually read_parquet_safe calls pandas first.
        raise e


def process_dataset_vllm(
    input_root,
    output_root,
    process_func,
    file_suffix="base.parquet",
    output_suffix=".jsonl",
    batch_size=1,
):
    """
    Main processing loop for VLLM tasks.

    Iterates over files in the input directory, reads them (Parquet or JSONL),
    batches records, applies a processing function, and writes results to the output directory
    while maintaining the directory structure.

    Args:
        input_root (str): Root directory containing input files (parquet or jsonl).
        output_root (str): Root directory for output files.
        process_func (callable): Function that takes a list of records and returns a list of results.
                                 Signature: process_func(records: List[dict]) -> List[dict]
                                 The returned dict should contain 'id' and result fields.
        file_suffix (str): Suffix of input files to look for (e.g., 'base.parquet', '.jsonl').
        output_suffix (str): Suffix for output files (default '.jsonl').
        batch_size (int): Number of records to accumulate before calling process_func.
    """
    input_files = find_files(input_root, suffix=file_suffix)
    print(f"Found {len(input_files)} files with suffix '{file_suffix}'.")

    current_output_path = None
    output_handle = None

    batch_records = []
    batch_meta_info = []  # Store (output_path, original_record) for the batch

    total_files = len(input_files)

    for i, input_path in enumerate(input_files):
        print(f"[{i + 1}/{total_files}] Reading {input_path}...")

        # Determine output path for this input file
        target_output_path = get_output_path(
            input_path, input_root, output_root, output_suffix
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(target_output_path), exist_ok=True)

        # Read records based on file type
        records = []
        try:
            if input_path.endswith(".parquet"):
                parquet_file = pq.ParquetFile(input_path)
                for batch in parquet_file.iter_batches():
                    df = batch.to_pandas()
                    records.extend(df.to_dict("records"))
            elif input_path.endswith(".jsonl"):
                with open(input_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {input_path}: {e}")
            continue

        for record in records:
            # Inject source file path for downstream processing (e.g. finding sibling files)
            record["_source_file_path"] = input_path

            batch_records.append(record)
            batch_meta_info.append(
                {"output_path": target_output_path, "record": record}
            )

            if len(batch_records) >= batch_size:
                _flush_batch(batch_records, batch_meta_info, process_func)
                batch_records = []
                batch_meta_info = []

    # Process remaining
    if batch_records:
        _flush_batch(batch_records, batch_meta_info, process_func)


def _flush_batch(records, meta_info, process_func):
    """
    Helper to process a batch and write results to correct files.
    """
    if not records:
        return

    try:
        results = process_func(records)
    except Exception as e:
        print(f"Error in process_func: {e}")
        return

    # Map by ID for safety
    result_map = {}
    for res in results:
        if "id" in res:
            result_map[res["id"]] = res

    # Group writes by output path to minimize file open/closes
    writes_by_path = {}

    for i, meta in enumerate(meta_info):
        rec_id = meta["record"].get("id")
        output_path = meta["output_path"]

        result = None
        if rec_id in result_map:
            result = result_map[rec_id]
        elif i < len(results):
            # Fallback to index if ID not found (risky if async/shuffled)
            # But for VLLM usually synchronous batch processing preserves order
            result = results[i]

        if result:
            if output_path not in writes_by_path:
                writes_by_path[output_path] = []
            writes_by_path[output_path].append(result)


def _type_bucket(v):
    if v is None:
        return "null"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, (list, tuple)):
        return "list"
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "bytes"
    return "scalar"


def _coerce_json_dict(value):
    """Ensure value is a dict; parse JSON strings; wrap scalars as {'_raw': ...}."""
    import json

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
            return {"_raw": obj}
        except Exception:
            return {"_raw": value}
    if isinstance(value, (list, tuple)):
        return {"_raw": list(value)}
    return {"_raw": value}


def _coerce_audio(value):
    """Normalize audio field to a dict with optional keys bytes/ark/offset/path."""
    if value is None:
        return {}
    if isinstance(value, dict):
        # best-effort keep existing
        out = {}
        if "bytes" in value:
            out["bytes"] = value.get("bytes")
        # common variants
        if "ark" in value:
            out["ark"] = value.get("ark")
        if "offset" in value:
            out["offset"] = value.get("offset")
        if "path" in value:
            out["path"] = value.get("path")
        return out or value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes": bytes(value)}
    if isinstance(value, str):
        return {"path": value}
    return {"_raw": value}


def _json_stringify(value):
    import json

    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def sanitize_records_for_arrow(records):
    """Make record list Arrow-writable without dropping rows.

    Strategy:
    - Force known struct-like columns ('meta','translation','audio') into dict.
    - For any other column where non-null values mix incompatible buckets
      (e.g. dict+scalar, list+scalar), coerce all non-null values to JSON strings.
    """
    import pyarrow as pa

    if not records:
        return records

    keys = set()
    for r in records:
        keys.update(r.keys())

    # First pass: bucket types per key
    buckets_by_key = {k: set() for k in keys}
    for r in records:
        for k in keys:
            if k in r:
                b = _type_bucket(r.get(k))
                if b != "null":
                    buckets_by_key[k].add(b)

    force_dict_keys = {"meta", "translation"}
    if "audio" in keys:
        force_dict_keys.add("audio")

    stringify_keys = set()
    for k, bs in buckets_by_key.items():
        if not bs or len(bs) == 1:
            continue
        if k in force_dict_keys:
            continue
        stringify_keys.add(k)

    out_records = []
    for r in records:
        nr = {}
        # Normalize pyarrow scalars (and other scalars exposing as_py)
        for k, v in r.items():
            if isinstance(v, pa.Scalar):
                try:
                    nr[k] = v.as_py()
                except Exception:
                    nr[k] = str(v)
            else:
                nr[k] = v

        for k in force_dict_keys:
            if k in nr:
                if k == "audio":
                    nr[k] = _coerce_audio(nr.get(k))
                else:
                    nr[k] = _coerce_json_dict(nr.get(k))

        for k in stringify_keys:
            if k in nr:
                nr[k] = _json_stringify(nr.get(k))

        out_records.append(nr)

    return out_records


def _find_problem_keys_for_arrow(records):
    import pyarrow as pa

    if not records:
        return []

    keys = set()
    for r in records:
        keys.update(r.keys())

    problem = []
    for k in keys:
        values = [r.get(k) for r in records]
        if all(v is None for v in values):
            continue
        try:
            pa.array(values)
        except pa.lib.ArrowInvalid:
            problem.append(k)
        except Exception:
            problem.append(k)

    return sorted(set(problem))


def stringify_keys_in_records(records, keys):
    if not records or not keys:
        return records
    keyset = set(keys)
    out = []
    for r in records:
        nr = dict(r)
        for k in keyset:
            if k in nr:
                nr[k] = _json_stringify(nr.get(k))
        out.append(nr)
    return out


def write_parquet_safe(records, output_path, schema=None, compression="zstd"):
    """
    Safely writes a list of dicts to a parquet file.
    Handles schema inference and common type issues by falling back to Pandas if PyArrow strict conversion fails.

    Critical: Sanitizes NaN values (float NaN) to None before writing, as PyArrow's from_pylist
    will fail with ArrowTypeError when encountering NaN in object/bytes columns.
    This commonly occurs when converting from pandas DataFrame with NaN values.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    import json
    import numpy as np
    import math

    if not records:
        print(f"Warning: No records to write to {output_path}")
        return

    # CRITICAL: Sanitize NaN values to None to avoid ArrowTypeError
    # "Expected bytes, got a 'float' object" occurs when pandas NaN values
    # exist in object/bytes columns (common after DataFrame operations)
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, float) and math.isnan(v):
                r[k] = None

    try:
        # Try direct pyarrow conversion first (fastest, strict)
        table = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(table, output_path, compression=compression)
    except Exception as e_arrow:
        # Strategy: Use advanced sanitization from repack_final_selection
        try:
            fixed = sanitize_records_for_arrow(records)
            table = pa.Table.from_pylist(fixed)
            pq.write_table(table, output_path, compression=compression)
            return
        except Exception as e_sanitize:
            # Final fallback: detect remaining problematic columns and stringify them
            try:
                problem_keys = _find_problem_keys_for_arrow(records)
                fixed2 = stringify_keys_in_records(records, problem_keys)
                table = pa.Table.from_pylist(fixed2)
                pq.write_table(table, output_path, compression=compression)
            except Exception as e_final:
                print(f"[ERROR] Failed to write parquet safely to {output_path}")
                print(f"  PyArrow error: {e_arrow}")
                print(f"  Sanitization error: {e_sanitize}")
                print(f"  Final fallback error: {e_final}")
                raise e_final
