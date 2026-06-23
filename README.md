# STEB Automatic Evaluation

This repository contains the automatic evaluation pipeline for Speech-to-Speech
Translation outputs used by STEB.

It includes:

- benchmark/result joining into `EvalRecord` JSONL
- hypothesis audio feature extraction for ASR, captions, emotion/style summary,
  and sound events
- automatic scoring for BLEU, duration/SLC, COMET/XCOMET, speaker similarity,
  and LLM emotion/style/event judges
- three-run LLM judge aggregation with the `v4` outlier-aware rule

## Setup

Create the main environment with uv:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

The caption and LLM judge steps expect OpenAI-compatible vLLM servers. The
default local launchers use:

- `QWEN3_CAPTION_MODEL_PATH` for the Qwen3-Omni captioner
- `QWEN3_INSTRUCT_MODEL_PATH` for the Qwen3 instruct model

Set these variables if your model paths differ from the defaults in
`core_functional_modules/start_servers.sh`.

Speaker similarity runs in an isolated environment because it uses the
Seed-TTS-eval-compatible UniSpeech stack:

```bash
uv venv .envs/speaker-sim --python 3.10
.envs/speaker-sim/bin/python -m pip install \
  torch torchaudio tqdm soundfile librosa packaging omegaconf \
  s3prl==0.3.1 fairseq==0.12.2

export SPEAKER_SIM_PYTHON=$PWD/.envs/speaker-sim/bin/python
export SPEAKER_SIM_CKPT=/path/to/wavlm_large_finetune.pth
```

## Input

Benchmark JSONL rows should contain source/reference fields:

```json
{
  "id": "sample_001",
  "text": "source transcript",
  "text_with_events": "source transcript [Laughter]",
  "translation": {"en": "reference translation"},
  "emotion": "cheerful",
  "style": "audiobook narration",
  "caption": "reference audio caption",
  "wav_path": "/path/to/reference.wav"
}
```

Result JSONL rows should contain hypothesis fields:

```json
{
  "id": "sample_001",
  "hyp_text": "Translation with sound events: ...",
  "hyp_wav_path": "/path/to/hypothesis.wav",
  "model_name": "my_model",
  "error": null
}
```

## Run

```bash
BENCHMARK_FILE=/path/to/benchmark.jsonl \
RESULTS_FILE=/path/to/results.jsonl \
OUTPUT_DIR=/path/to/eval_output \
SPLIT=normal \
SRC_LANG=zh \
TGT_LANG=en \
ASR_MODEL_PATH=/path/to/Qwen3-ASR-1.7B \
ALIGNER_MODEL_PATH=/path/to/Qwen3-ForcedAligner-0.6B \
ENABLE_LLM=--enable_llm \
bash evaluation/run_eval.sh
```

For event-bearing samples, set `SPLIT=event`. Phase 2 will run BEATs sound
event detection and event combination before scoring.

Useful flags:

- `START_PHASE=3 END_PHASE=3` runs scoring only from existing eval records.
- `ENABLE_COMET=--enable_comet` enables COMET.
- `ENABLE_XCOMET=--enable_xcomet` enables XCOMET.
- `ENABLE_SPEAKER_SIM=--enable_speaker_sim` enables speaker similarity.
- `AUTO_START_CAPTION_SERVERS=0 CAPTION_SERVER_URLS=http://host:port/v1`
  uses manually launched caption servers.
- `AUTO_START_INSTRUCT_SERVERS=0 INSTRUCT_SERVER_URLS=http://host:port/v1`
  uses manually launched instruct servers.

Outputs are written under `OUTPUT_DIR`, including:

- `eval_records.jsonl`
- `eval_records_merged.jsonl`
- `eval_results_<model>.jsonl`
- `eval_summary_<model>.json`
- per-phase logs under `logs/`

## LLM Judge

The LLM judge prompt version is `v4_choice`.

The default LLM ensemble strategy is `v4`:

- if two of three scores agree and the third differs by at least 2, keep the
  agreed score
- if all three scores differ, use the median
- otherwise use the mean
