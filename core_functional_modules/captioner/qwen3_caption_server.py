import argparse
import os
import sys
import json
import asyncio
import torch
import torchaudio
import base64
from io import BytesIO

# Add utils to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
utils_dir = os.path.join(project_root, 'utils')
sys.path.append(utils_dir)

from vllm_client import VLLMClient, process_dataset_async

CAPTION_PROMPT = (
    "Describe this speech audio for evaluation. Focus on vocal emotion, "
    "speaking style, scene or genre, and delivery cues such as pace, energy, "
    "formality, and expressiveness. Do not infer emotion only from the words. "
    "Keep the caption concise and useful for comparing two speech clips."
)

def process_audio(audio_bytes, target_sr=16000):
    try:
        waveform, sr = torchaudio.load(BytesIO(audio_bytes))
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wavform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform, target_sr
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None

async def caption_record_func(client, session, record):
    audio_id = record.get('id', 'unknown')
    
    # Extract meta info to save
    meta = record.copy()
    if 'audio' in meta:
        del meta['audio']
    
    # Get audio bytes
    audio_data = record.get('audio')
    if isinstance(audio_data, dict) and 'bytes' in audio_data:
        audio_bytes = audio_data['bytes']
    elif isinstance(audio_data, bytes):
            audio_bytes = audio_data
    else:
        return None
    
    # Process Audio
    waveform, sr = process_audio(audio_bytes)
    if waveform is None:
        return None
    
    try:
        # Convert to base64 directly without saving to disk
        buffer = BytesIO()
        torchaudio.save(buffer, waveform, sr, format="wav")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Construct Message with Data URI
        audio_url = f"data:audio/wav;base64,{audio_base64}"
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "audio_url", "audio_url": {"url": audio_url}}
                ]
            }
        ]
        
        response = await client.chat_completions(messages, session)
        
        if response and 'choices' in response:
            caption = response['choices'][0]['message']['content']
            return {
                "id": audio_id,
                "caption": caption,
                "meta": meta
            }
        else:
            print(f"Failed to get caption for {audio_id}")
            return None

    except Exception as e:
        print(f"Error generating caption for {audio_id}: {e}")
        return None

def main(args):
    client = VLLMClient(base_url=args.server_url, model_name=args.model_name)
    
    asyncio.run(process_dataset_async(
        input_root=args.parquet_path,
        output_root=args.output_root,
        client=client,
        process_record_func=caption_record_func,
        file_suffix='base.parquet', # Match original script behavior
        output_suffix='.jsonl',
        concurrency=args.concurrency,
        batch_write_size=args.batch_size
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Captioning Server Client')
    parser.add_argument('-i', '--parquet_path', type=str, required=True, help='Root directory containing parquet files')
    parser.add_argument('-o', '--output_root', type=str, required=True, help='Root directory for output jsonl files')
    parser.add_argument('--server_url', type=str, default="http://localhost:8901/v1", help='VLLM Server URL')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-Omni-30B-A3B-Captioner", help='Model name')
    parser.add_argument('--concurrency', type=int, default=40, help='Number of concurrent requests')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of records to write at once')
    
    args = parser.parse_args()
    main(args)
