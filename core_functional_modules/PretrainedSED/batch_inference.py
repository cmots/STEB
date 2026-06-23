import argparse
import librosa
import torch
import pandas as pd
import os
import sys
import shutil
import random
from tqdm import tqdm
import pyarrow.parquet as pq
import torchaudio
from io import BytesIO
import json

# Add project root to sys.path to allow importing from utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import parquet_io as parquet_io2

from data_util import audioset_classes
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper

def load_model(args, device):
    model_name = args.model_name
    if model_name == "BEATs":
        beats = BEATsWrapper()
        model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")
    else:
        raise NotImplementedError("The public STEB eval pipeline supports model_name='BEATs'.")

    model.eval()
    model.to(device)
    return model

def find_parquet_files(root_path):
    files = parquet_io2.find_files(root_path, suffix='base.parquet')
    return sorted(files)  # Ensure deterministic order for distributed processing

def get_output_path(parquet_path, input_root, output_root):
    # Calculate relative path to mirror structure
    # If input_root is /data and parquet_path is /data/a/b.parquet
    # rel_path is a/b.parquet
    # output will be output_root/a/b.jsonl
    
    if os.path.isfile(input_root):
        rel_path = os.path.basename(parquet_path)
    else:
        rel_path = os.path.relpath(parquet_path, input_root)
    
    rel_path_no_ext = os.path.splitext(rel_path)[0]
    jsonl_path = os.path.join(output_root, f"{rel_path_no_ext}.jsonl")
    return jsonl_path

def parquet_chunk_generator(parquet_paths, device, processed_ids=None):
    sample_rate = 16000
    segment_duration = 10
    segment_samples = segment_duration * sample_rate

    for parquet_path in parquet_paths:
        try:
            print(f"Processing {parquet_path}...")
            records = parquet_io2.iter_parquet_safe(parquet_path)
        except Exception as e:
            print(f"Error opening parquet file {parquet_path}: {e}")
            continue

        for row in tqdm(records, desc=f"Processing records in {os.path.basename(parquet_path)}"):
            try:
                audio_id = row['id']
                if processed_ids is not None and audio_id in processed_ids:
                    continue

                if isinstance(row.get('audio'), dict) and 'bytes' in row['audio']:
                    audio_bytes = row['audio']['bytes']
                elif 'bytes' in row:
                    audio_bytes = row['bytes']
                else:
                    print(f"Unexpected audio format for id {audio_id}")
                    continue

                # load audio
                waveform, sr = torchaudio.load(BytesIO(audio_bytes))
                
                # Resample
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)
                
                # Mix to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                waveform = waveform.to(device)
                waveform_len = waveform.shape[1]
                audio_len = waveform_len / sample_rate

                # split audio file into 10-second chunks
                num_chunks = waveform_len // segment_samples + (1 if waveform_len % segment_samples != 0 else 0)

                for i in range(num_chunks):
                    start_idx = i * segment_samples
                    end_idx = min((i + 1) * segment_samples, waveform_len)
                    waveform_chunk = waveform[:, start_idx:end_idx]

                    # Pad the last chunk if it's shorter than 10 seconds
                    if waveform_chunk.shape[1] < segment_samples:
                        pad_size = segment_samples - waveform_chunk.shape[1]
                        waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, pad_size))

                    yield {
                        'id': audio_id,
                        'chunk': waveform_chunk,
                        'chunk_idx': i,
                        'total_chunks': num_chunks,
                        'duration': audio_len
                    }
            except Exception as e:
                print(f"Error processing {row.get('id', 'unknown')}: {e}")
                continue

def main(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    model = load_model(args, device)
    
    # Check execution mode
    is_dir_mode = os.path.isdir(args.parquet_path)
    if is_dir_mode and not os.path.exists(args.output_jsonl):
        # Allow creating output dir if it doesn't exist
        try:
            os.makedirs(args.output_jsonl, exist_ok=True)
        except:
             pass

    parquet_files = find_parquet_files(args.parquet_path)
    print(f"Found {len(parquet_files)} parquet files total.")
    
    # Randomize file order to reduce contention in distributed locking
    random.shuffle(parquet_files)

    # Storage for aggregating results per file
    # key: id, value: {'preds': [None]*total, 'duration': dur, 'count': 0, 'out_handle': handle}
    file_results = {}
    
    # Current Output State
    current_output_handle = None
    current_output_path = None

    def get_output_handle(parquet_path):
        nonlocal current_output_handle, current_output_path
        
        if is_dir_mode:
            target_path = get_output_path(parquet_path, args.parquet_path, args.output_jsonl)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
        else:
            target_path = args.output_jsonl
            
        if target_path == current_output_path and current_output_handle is not None:
            return current_output_handle, None 

        if current_output_handle:
            current_output_handle.close()
            
        current_output_path = target_path
        
        processed_ids = set()
        output_mode = 'w'

        if os.path.exists(target_path):
            print(f"Checking existing output {target_path}...")
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                rec = json.loads(line)
                                if 'id' in rec:
                                    processed_ids.add(rec['id'])
                            except:
                                pass
                print(f"Found {len(processed_ids)} already processed IDs in {os.path.basename(target_path)}.")
            except Exception as e:
                print(f"Error reading existing file: {e}")

            if len(processed_ids) > 0:
                output_mode = 'a'
            else:
                output_mode = 'w'
        
        current_output_handle = open(target_path, output_mode, encoding='utf-8')
        return current_output_handle, processed_ids

    def finalize_file(audio_id, data, out_handle):
        try:
            y_strong = torch.cat(data['preds'], dim=1) # (C, Total_T)
            y_strong = y_strong.unsqueeze(0) # (1, C, Total_T)
            y_strong = torch.sigmoid(y_strong)

            encoder = ManyHotEncoder(audioset_classes.as_strong_train_classes, audio_len=data['duration'])

            (
                scores_unprocessed,
                scores_postprocessed,
                decoded_predictions
            ) = batched_decode_preds(
                y_strong.float(),
                [audio_id],
                encoder,
                median_filter=args.median_window,
                thresholds=args.detection_thresholds,
            )

            result = {
                "id": audio_id,
                "events": {}
            }
            
            for th in decoded_predictions:
                events_df = decoded_predictions[th]
                events_list = events_df.to_dict(orient='records')
                result["events"][str(th)] = events_list
            
            json.dump(result, out_handle, ensure_ascii=False)
            out_handle.write('\n')
            out_handle.flush()

        except Exception as e:
            print(f"Error finalizing {audio_id}: {e}")

    def process_batch(tensors, metas):
        if not tensors:
            return
        
        batch_input = torch.cat(tensors, dim=0) # (B, 1, Samples)
        
        with torch.no_grad():
            mel = model.mel_forward(batch_input)
            y_strong, _ = model(mel) # (B, C, T)
        
        for i, pred in enumerate(y_strong):
            meta = metas[i]
            audio_id = meta['id']
            
            if audio_id not in file_results:
                file_results[audio_id] = {
                    'preds': [None] * meta['total_chunks'],
                    'duration': meta['duration'],
                    'count': 0,
                    'total': meta['total_chunks'],
                    'out_handle': current_output_handle
                }
            
            file_results[audio_id]['preds'][meta['chunk_idx']] = pred
            file_results[audio_id]['count'] += 1
            
            if file_results[audio_id]['count'] == file_results[audio_id]['total']:
                finalize_file(audio_id, file_results[audio_id], file_results[audio_id]['out_handle'])
                del file_results[audio_id]

    batch_tensors = []
    batch_meta = []

    for p_file in parquet_files:
        lock_dir = None
        done_marker = None
        has_lock = False
        
        # Dynamic Locking & Skipping Logic
        if is_dir_mode:
            target_path = get_output_path(p_file, args.parquet_path, args.output_jsonl)
            lock_dir = target_path + ".lock"
            done_marker = target_path + ".done"
            
            # 1. Check if already marked done
            if os.path.exists(done_marker):
                continue
                
            # 2. Try to acquire lock
            try:
                os.makedirs(lock_dir)
                has_lock = True
            except OSError:
                # Locked by another worker -> skip
                continue
        
        try:
            try:
               out_handle, processed_ids = get_output_handle(p_file)
            except Exception as e:
               print(f"Error setting up output for {p_file}: {e}")
               continue
            
            if len(processed_ids) > 0:
                print(f"Skipping {len(processed_ids)} processed records in {os.path.basename(p_file)}")

            gen = parquet_chunk_generator([p_file], device, processed_ids=processed_ids)
            
            file_has_data = False
            all_skipped = True
            
            for item in gen:
                file_has_data = True
                all_skipped = False
                batch_tensors.append(item['chunk'])
                batch_meta.append(item)
                
                if len(batch_tensors) >= args.batch_size:
                    process_batch(batch_tensors, batch_meta)
                    batch_tensors = []
                    batch_meta = []
            
            # End of file - flush batch to ensure no boundary crossing for safety
            if batch_tensors:
                process_batch(batch_tensors, batch_meta)
                batch_tensors = []
                batch_meta = []
            
            # Mark as done only if we skipped everything (meaning complete run before)
            # OR if we processed data successfully.
            if is_dir_mode and (all_skipped or file_has_data) and has_lock:
                 with open(done_marker, 'w') as f:
                     f.write('done')

        finally:
            # Release lock
            if has_lock and lock_dir and os.path.exists(lock_dir):
                try:
                    os.rmdir(lock_dir)
                except:
                    pass
            
    if current_output_handle:
        current_output_handle.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch inference for SED')
    parser.add_argument('--model_name', type=str, default='BEATs')
    parser.add_argument('--parquet_path', type=str, required=True, help='Path to input parquet file or directory')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to output jsonl file or directory')
    parser.add_argument('--detection_thresholds', type=float, nargs='+', default=[0.1, 0.2, 0.5])
    parser.add_argument('--median_window', type=float, default=9)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing chunks')
    args = parser.parse_args()

    assert args.model_name in ["BEATs", "ASIT", "ATST-F", "fpasst", "M2D"] or args.model_name.startswith("frame_mn")
    main(args)
