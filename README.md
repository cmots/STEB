# STEB

Official code release for **STEB**, an automatic evaluation toolkit for
speech-to-speech translation systems. The repository provides the evaluation
pipeline, feature extraction modules, and metric implementations used to score
translation quality, speech preservation, and paralinguistic consistency.

<p align="center">
  <b>Paper</b> coming soon |
  <b>Homepage</b> https://cmots.github.io/steb.github.io/ |
  <b>Hugging Face</b> https://huggingface.co/datasets/cmots/STEB
</p>

## Overview

STEB evaluates speech-to-speech translation outputs from both text and audio
signals. Given benchmark metadata, reference annotations, and model-generated
hypothesis audio, the pipeline builds unified evaluation records, extracts
hypothesis-side audio features, and reports sentence-level and corpus-level
scores.

The current release focuses on the automatic evaluation code and provides a
`uv`-based setup for the default vLLM-backed evaluation workflow. Optional
COMET/XCOMET and speaker-similarity workflows require isolated environments
because their upstream dependencies conflict with the modern vLLM stack.

## Highlights

- Data joining for benchmark and model result JSONL files.
- Hypothesis audio packing into Parquet for batched feature extraction.
- Qwen3-ASR based transcription and word-level timestamp extraction.
- Qwen3-Omni based audio captioning and Qwen3-Instruct based emotion/style
  summarization through OpenAI-compatible vLLM servers.
- BEATs-based sound event detection and event tag insertion.
- BLEU, duration/SLC, COMET/XCOMET, speaker similarity, and LLM judge scorers.
- Robust repeated judging for LLM-based metrics.

## Repository Structure

```text
STEB/
|-- core_functional_modules/
|   |-- captioner/              # Caption and emotion/style feature clients
|   |-- extract_timestamp/      # ASR timestamp extraction and event merging
|   |-- PretrainedSED/          # BEATs sound event detection wrapper
|   `-- utils/                  # Parquet, task, and vLLM client utilities
|-- evaluation/
|   |-- eval/                   # Data loading, feature merging, and scorers
|   |-- run_eval.sh             # End-to-end evaluation entry point
|   `-- service_orchestrator.sh # Local vLLM service lifecycle helpers
|-- requirements.txt
`-- README.md
```

## Installation

STEB is configured as a `uv` project. Public users do not need Conda or any
machine-specific base environment. From a fresh clone, install `uv` following
the official instructions, then let `uv` create and manage the project
environment:

```bash
uv sync --python 3.10
uv run python -c "import torch, torchaudio, vllm, pandas, sacrebleu; print('STEB environment OK')"
```

The default environment includes `vllm==0.23.0`, `qwen-omni-utils`, and the
BEATs / PretrainedSED runtime dependencies used for non-verbal (NV) sound-event
extraction, because the default evaluation keeps BLEU, emotion, style, NV, and
SLC enabled while leaving speaker similarity and COMET/XCOMET disabled unless
explicitly run in their optional environments.

For platforms that need a specific PyTorch wheel, install the matching PyTorch
build first with `uv pip install`, then synchronize the remaining project
dependencies. Choose the index URL from the official PyTorch installation
selector for your operating system, Python version, and accelerator:

```bash
uv venv --python 3.10
uv pip install torch torchaudio --index-url <pytorch-wheel-index-url>
uv sync
```

The project dependencies are maintained with `uv add`. You only need these
commands when changing STEB's dependency set, not for normal installation:

```bash
uv init --bare --python 3.10
uv add aiohttp dcase-util librosa numpy pandas pyarrow qwen-omni-utils requests sacrebleu scipy sed-scores-eval soundfile torch torchaudio tqdm transformers vllm==0.23.0 zhconv zhon
uv add --optional speaker-sim fairseq==0.12.2 fire omegaconf packaging s3prl==0.3.1
```

Install the optional speaker-sim dependency group only if you want to build the
isolated speaker-sim environment from the project metadata:

```bash
uv sync --extra speaker-sim
```

### Optional COMET / XCOMET scoring

COMET and XCOMET are disabled by default and are not listed as project extras,
because `unbabel-comet` cannot be resolved in the same `uv` project environment
as `vllm==0.23.0`. Keep COMET in a separate `uv` environment and run only Phase
3 COMET scoring there:

```bash
uv venv .envs/comet --python 3.10
uv pip install --python .envs/comet/bin/python \
  unbabel-comet zhconv zhon sacrebleu tqdm soundfile torchaudio

.envs/comet/bin/python -m evaluation.eval.run_full_eval \
  --input /path/to/eval_records_merged.jsonl \
  --output_dir /path/to/eval_output \
  --src_lang zh --tgt_lang en \
  --enable_comet \
  --base_comet_model Unbabel/wmt22-comet-da
```

For XCOMET, install the same isolated environment and pass an XCOMET model ID or
local checkpoint path with `--enable_xcomet --xcomet_model <model-or-path>`.
Model names may be Hugging Face IDs; `unbabel-comet` will download them on first
use, or you can pass a local checkpoint/cache directory.

### Optional speaker similarity scoring

Speaker similarity follows the Seed-TTS-eval-compatible UniSpeech stack and is
kept in an isolated environment because of its pinned legacy dependencies:

```bash
uv venv .envs/speaker-sim --python 3.10
uv pip install --python .envs/speaker-sim/bin/python \
  torch torchaudio tqdm soundfile librosa packaging omegaconf \
  s3prl==0.3.1 fairseq==0.12.2 fire

export SPEAKER_SIM_PYTHON=$PWD/.envs/speaker-sim/bin/python
export SPEAKER_SIM_CKPT=/path/to/wavlm_large_finetune.pth
```

Download `wavlm_large_finetune.pth` before enabling speaker similarity:

```text
https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view
```

When `ENABLE_SPEAKER_SIM=--enable_speaker_sim` is set, `evaluation/run_eval.sh`
checks both `SPEAKER_SIM_PYTHON` and `SPEAKER_SIM_CKPT` before scoring.
Internally, STEB loads the vendored UniSpeech speaker-verification model
(`wavlm_large` + ECAPA-TDNN), resamples the hypothesis and reference wav files to
16 kHz, extracts one speaker embedding per file, and reports
`speaker_similarity` as the cosine similarity between the two embeddings.

For legacy workflows, `requirements.txt` remains available:

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

## Data Format

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

## Running Evaluation

The main entry point is `evaluation/run_eval.sh`. All dataset, output, and
model paths are configured through environment variables:

```bash
BENCHMARK_FILE=/path/to/benchmark.jsonl \
RESULTS_FILE=/path/to/results.jsonl \
OUTPUT_DIR=/path/to/eval_output \
SPLIT=normal \
SRC_LANG=zh \
TGT_LANG=en \
ASR_MODEL_PATH=/path/to/Qwen3-ASR-1.7B \
ALIGNER_MODEL_PATH=/path/to/Qwen3-ForcedAligner-0.6B \
QWEN3_CAPTION_MODEL_PATH=/path/to/Qwen3-Omni-Captioner \
QWEN3_INSTRUCT_MODEL_PATH=/path/to/Qwen3-Instruct \
ENABLE_LLM=--enable_llm \
bash evaluation/run_eval.sh
```

For event-bearing samples, set `SPLIT=event`. Phase 2 will run BEATs sound
event detection and event combination before scoring.

Useful switches:

- `START_PHASE=3 END_PHASE=3` runs scoring only from existing eval records.
- `ENABLE_COMET=--enable_comet BASE_COMET_MODEL=/path/or/hf/id` enables optional
  COMET scoring from a separate COMET environment.
- `ENABLE_XCOMET=--enable_xcomet XCOMET_MODEL=/path/or/hf/id` enables optional
  XCOMET scoring from a separate COMET environment.
- `ENABLE_SPEAKER_SIM=--enable_speaker_sim` enables optional speaker similarity
  scoring from the isolated speaker-sim environment.
- `AUTO_START_CAPTION_SERVERS=0 CAPTION_SERVER_URLS=http://host:port/v1` uses
  manually launched caption servers.
- `AUTO_START_INSTRUCT_SERVERS=0 INSTRUCT_SERVER_URLS=http://host:port/v1` uses
  manually launched instruct servers.

Outputs are written under `OUTPUT_DIR`:

- `eval_records.jsonl`
- `eval_records_merged.jsonl`
- `eval_results_<model>.jsonl`
- `eval_summary_<model>.json`
- per-phase logs under `logs/`

## Metrics

The default pipeline reports:

- **Text translation:** BLEU.
- **Speech timing:** duration ratio and SLC.
- **Paralinguistic consistency:** LLM emotion, style, and non-verbal (NV) sound-event judges.

Optional environments can additionally report:

- **Text translation:** COMET/XCOMET.
- **Speaker preservation:** UniSpeech speaker similarity.

## Citation

Citation information will be added with the paper link.

## Acknowledgements

This repository includes or interfaces with components from vLLM, Qwen,
PretrainedSED/BEATs, UniSpeech, SacreBLEU, and COMET. Please also follow the
licenses and usage terms of the corresponding upstream projects and model
checkpoints.

## License

Project-level license information will be added before public release.
Third-party code in subdirectories retains its upstream license.
