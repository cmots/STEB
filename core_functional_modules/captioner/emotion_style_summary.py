import argparse
import os
import sys
import json
import asyncio
import re

# Add utils to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
utils_dir = os.path.join(project_root, 'utils')
sys.path.append(utils_dir)

from vllm_client import VLLMClient, process_dataset_async

# ==========================================
# PROMPT TEMPLATE
# ==========================================
PROMPT_TEMPLATE = '''
You are an expert audio analyst specializing in paralinguistics and voice psychology.
Your task is to analyze a raw audio caption and extract the **Emotion** and **Style** of the speech.

**Input Analysis Rules:**
1.  **Ignore the Transcription**: Do not judge the emotion only based on *what* is said (the words). You must judge based on *how* it is said.
2.  **Focus on Vocal Delivery**: Look for cues like pitch, tempo, volume, breathing, and timbre in the caption.
3.  **Infer, Don't Just Describe**: Do not just list the acoustic features (e.g., "high pitch"). Instead, think what emotional state those features indicate.

**Output Requirements:**
- Output strictly in **JSON format**.
- The values for "emotion" and "style" must be **short, descriptive, natural sentences** (avoiding complex or obscure words).
- Pure Emotion Only: The "emotion" field must describe the *feeling* or *attitude* (e.g., "The speaker is feeling deeply sorrowful"). **DO NOT** mention the physical acoustic traits (e.g., **DO NOT** say "because the voice is low" or "due to the slow tempo").

**JSON Attributes:**
- **"emotion"**: A sentence describing the speaker's internal feeling or attitude, derived from their vocal cues.
- **"style"**: A sentence describing the scenario, genre, or context of the speech (e.g., news, ad, casual chat, movie acting, interview, living, tutorial, etc.).

**Input Caption:**
"""
{caption}
"""

**JSON Schema:**
{{
  "emotion": "A short sentence describing the vocal emotion.",
  "style": "A short sentence describing the speaking scenario."
}}
'''


def build_prompt(caption, prompt_version=None):
    if prompt_version:
        raise ValueError("The public STEB summary step keeps a single prompt template.")
    return PROMPT_TEMPLATE.format(caption=caption)


def extract_json(text):
    try:
        # Try to find JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return json.loads(text)
    except Exception as e:
        print(f"JSON extraction failed: {e}")
        return None

async def emotion_style_record_func(client, session, record):
    try:
        caption = record.get('caption', '') or record.get('text', '')
        if not caption:
            return None
        
        prompt = build_prompt(caption)
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await client.chat_completions(messages, session)
        
        if response and 'choices' in response:
            content = response['choices'][0]['message']['content']
            parsed = extract_json(content)
            
            result = {
                "id": record.get('id'),
                "original_caption": caption,
                "raw_output": content,
                "meta": record.get('meta', {})
            }
            
            if parsed:
                result["emotion"] = parsed.get("emotion")
                result["style"] = parsed.get("style")
            
            return result
        else:
            print(f"Request failed or empty response for id {record.get('id')}: {response}")
            return None
    except Exception as e:
        print(f"Error processing record {record.get('id')}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(args):
    client = VLLMClient(base_url=args.server_url, model_name=args.model_name)
    
    asyncio.run(process_dataset_async(
        input_root=args.input_jsonl,
        output_root=args.output_jsonl,
        client=client,
        process_record_func=emotion_style_record_func,
        file_suffix='.jsonl', # Usually input is jsonl from caption step
        output_suffix='.jsonl',
        concurrency=args.concurrency,
        batch_write_size=args.batch_size
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion and Style Summary Server Client')
    parser.add_argument('-i', '--input_jsonl', type=str, required=True, help='Path to input jsonl file or directory')
    parser.add_argument('-o', '--output_jsonl', type=str, required=True, help='Path to output jsonl file or directory')
    parser.add_argument('--server_url', type=str, default="http://localhost:8902/v1", help='VLLM Server URL(s), comma separated')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help='Model name')
    parser.add_argument('--concurrency', type=int, default=100, help='Number of concurrent requests')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of records to write at once')
    
    args = parser.parse_args()
    main(args)
