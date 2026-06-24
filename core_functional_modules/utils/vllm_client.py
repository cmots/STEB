import aiohttp
import asyncio
import json
import os
import sys
import time
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
import pyarrow.parquet as pq

# Add utils to sys.path to import parquet_io if needed for file finding
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from parquet_io import find_files, get_output_path, read_parquet_safe

import random

class VLLMClient:
    def __init__(
        self,
        base_url="http://localhost:8000/v1",
        model_name=None,
        api_key="EMPTY",
        max_tokens=None,
    ):
        # Support multiple URLs for load balancing
        if isinstance(base_url, str):
            self.base_urls = [url.strip() for url in base_url.split(',')]
        else:
            self.base_urls = base_url
            
        self.model_name = model_name
        self.api_key = api_key
        if max_tokens is None:
            max_tokens = os.environ.get("VLLM_CLIENT_MAX_TOKENS", "2048")
        self.max_tokens = int(max_tokens) if str(max_tokens).strip() else None
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # Track active requests per URL for Least Connections load balancing
        self.active_requests = {url: 0 for url in self.base_urls}

    def _get_best_url(self):
        # Least connections strategy
        # Find the url with minimum active requests
        min_active = min(self.active_requests.values())
        candidates = [url for url, count in self.active_requests.items() if count == min_active]
        return random.choice(candidates)

    async def chat_completions(self, messages, session, **kwargs):
        base_url = self._get_best_url()
        self.active_requests[base_url] += 1
        
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }
        if self.max_tokens is not None and "max_tokens" not in payload:
            payload["max_tokens"] = self.max_tokens
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        print(f"Error {response.status} from {base_url}: {text}")
                        # If it's a server error (5xx), we might want to retry
                        if 500 <= response.status < 600 and attempt < max_retries:
                            await asyncio.sleep(0.1 * (2 ** attempt)) # Exponential backoff
                            continue
                        return None
                    return await response.json()
            except Exception as e:
                print(f"Request failed to {base_url} (Attempt {attempt+1}/{max_retries+1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt)) # Exponential backoff
                    continue
                return None
            finally:
                if attempt == max_retries or (attempt < max_retries and 'response' in locals() and response.status == 200):
                     self.active_requests[base_url] -= 1

import itertools
from tqdm import tqdm

def load_file_data(input_path):
    records = []
    try:
        if input_path.endswith('.parquet'):
            records = read_parquet_safe(input_path)

        elif input_path.endswith('.jsonl'):
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Error decoding JSON at line {line_num+1} in {input_path}: {e}")
    except Exception as e:
        print(f"Error reading file {input_path}: {e}")
    return records

def get_processed_ids(output_path):
    """Helper to read existing output file and collect processed IDs."""
    ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # Check common ID fields
                            for key in ['id', 'index', 'key', 'msg_id']:
                                if key in data:
                                    ids.add(str(data[key]))
                                    break
                        except:
                            pass
        except Exception as e:
            print(f"Warning: Failed to read existing ids from {output_path}: {e}")
    return ids

async def process_dataset_async(
    input_root, 
    output_root, 
    client, 
    process_record_func, 
    file_suffix='base.parquet', 
    output_suffix='.jsonl', 
    concurrency=50,
    batch_write_size=100,
    **kwargs
):
    """
    Async version of process_dataset_vllm with streaming write.
    Supports global concurrency across multiple files.
    """
    input_files = find_files(input_root, suffix=file_suffix)
    print(f"Found {len(input_files)} files with suffix '{file_suffix}'.")
    
    # =================================================================
    # Global Worker Pool Implementation (Cross-File Concurrency)
    # =================================================================
    
    # Queue for distributing records to workers
    input_queue = asyncio.Queue(maxsize=concurrency * 4)
    
    # Queue for collecting results to write
    # Item format: (output_path, result_dict)
    result_queue = asyncio.Queue()
    
    # 1. Global Producer: Iterates files, reads data, and feeds into input_queue
    async def global_producer():
        loop = asyncio.get_running_loop()
        for i, input_path in enumerate(input_files):
            output_path = get_output_path(input_path, input_root, output_root, output_suffix)
            
            processed_ids = set()
            if os.path.exists(output_path):
                # Read existing IDs to skip them
                processed_ids = await loop.run_in_executor(None, get_processed_ids, output_path)
                if processed_ids:
                    print(f"Resuming {os.path.basename(output_path)}: Found {len(processed_ids)} processed records")
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Read Data in a separate thread
            records = await loop.run_in_executor(None, load_file_data, input_path)
            
            if not records:
                continue

            # Filter already processed records
            records_to_process = []
            if processed_ids:
                for record in records:
                    rec_id = None
                    for key in ['id', 'index', 'key', 'msg_id']:
                        if key in record:
                            rec_id = str(record[key])
                            break
                    
                    # If ID not found or not processed, keep it
                    if rec_id is None or rec_id not in processed_ids:
                        records_to_process.append(record)
            else:
                records_to_process = records

            if not records_to_process:
                print(f"Skipping {os.path.basename(input_path)} (All {len(records)} processed)")
                continue

            print(f"[{i+1}/{len(input_files)}] Queuing {len(records_to_process)}/{len(records)} records from {os.path.basename(input_path)}")
            
            for record in records_to_process:
                # Package everything needed for processing
                task_item = {
                    'record': record,
                    'output_path': output_path
                }
                await input_queue.put(task_item)
        
        # Signal workers to stop
        for _ in range(concurrency):
            await input_queue.put(None)

    # 2. Global Worker: Consumes records and processes them
    async def global_worker(session):
        while True:
            item = await input_queue.get()
            if item is None:
                input_queue.task_done()
                break
            
            try:
                record = item['record']
                output_path = item['output_path']
                
                res = await process_record_func(client, session, record, **kwargs)
                if res:
                    await result_queue.put((output_path, res))
            except Exception as e:
                print(f"Error processing record: {e}")
            finally:
                input_queue.task_done()

    # 3. Global Writer: Collects results and writes to files
    async def global_writer():
        # Cache open file handles: {output_path: (file_handle, buffer_list)}
        file_handles = {}
        
        while True:
            try:
                item = await asyncio.wait_for(result_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # Flush all buffers if no result comes in for 5 seconds
                for path, (fh, buf) in file_handles.items():
                    if buf:
                        try:
                            fh.write('\n'.join(buf) + '\n')
                            fh.flush()
                            buf.clear()
                        except Exception as e:
                            print(f"Error flushing to {path}: {e}")
                continue

            if item is None:
                result_queue.task_done()
                break
            
            output_path, res_dict = item
            
            if output_path not in file_handles:
                try:
                    fh = open(output_path, 'a', encoding='utf-8')
                    file_handles[output_path] = (fh, [])
                except Exception as e:
                    print(f"Error opening output file {output_path}: {e}")
                    result_queue.task_done()
                    continue
            
            fh, buffer = file_handles[output_path]
            buffer.append(json.dumps(res_dict, ensure_ascii=False))
            
            if len(buffer) >= batch_write_size:
                try:
                    fh.write('\n'.join(buffer) + '\n')
                    fh.flush()
                    buffer.clear()
                except Exception as e:
                    print(f"Error writing batch to {output_path}: {e}")
            
            result_queue.task_done()
        
        # Final cleanup
        for path, (fh, buf) in file_handles.items():
            if buf:
                try:
                    fh.write('\n'.join(buf) + '\n')
                    fh.flush()
                except Exception as e:
                    print(f"Error final write to {path}: {e}")
            fh.close()

    # Start all components
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        
        producer_task = asyncio.create_task(global_producer())
        worker_tasks = [asyncio.create_task(global_worker(session)) for _ in range(concurrency)]
        writer_task = asyncio.create_task(global_writer())
        
        await producer_task
        print("All files read and queued.")
        
        await input_queue.join()
        print("All tasks processed.")
        
        await asyncio.gather(*worker_tasks)
        
        await result_queue.put(None)
        await writer_task
        print("All results written.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick test for reading a parquet/jsonl file (focus on base.parquet audio).")
    parser.add_argument(
        "input_path",
        help="Path to parquet/jsonl to test reading",
    )
    args = parser.parse_args()

    # Direct single-batch probe to avoid full conversion issues
    recs = load_file_data(args.input_path)
    print(f"Read {len(recs)} records from {args.input_path}")
    if recs:
        first = recs[0]
        print("First keys:", list(first.keys()))
        if 'audio' in first:
            audio = first['audio']
            if isinstance(audio, dict) and 'bytes' in audio:
                print(f"Audio bytes length: {len(audio['bytes'])}")
        preview_keys = list(first.keys())[:3]
        print("First record preview:", {k: first[k] for k in preview_keys})
