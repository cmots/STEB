# STEB Code Structure

This file is a quick orientation guide for code agents working in this
repository. For user-facing setup details, read `README.md` first.

## Top-Level Layout

```text
STEB/
|-- core_functional_modules/   # Feature extraction and service clients
|-- evaluation/                # End-to-end evaluation orchestration and scorers
|-- pyproject.toml             # uv project metadata
|-- requirements.txt           # Legacy dependency list
|-- uv.lock                    # Locked uv dependency graph
`-- README.md                  # Public project README
```

The repository is script-oriented rather than packaged as an installable
library. `pyproject.toml` sets `package = false`.

## Main Entry Points

- `evaluation/run_eval.sh`
  End-to-end shell entry point. It joins benchmark/results JSONL, extracts
  hypothesis-side audio features, merges features back into evaluation records,
  and runs scoring.

- `evaluation/eval/run_full_eval.py`
  Python scoring orchestrator. It can run full scoring, metric-group scoring,
  shard scoring, and aggregation of partial group outputs.

- `core_functional_modules/start_servers.sh`
  Starts local OpenAI-compatible vLLM services for captioning and LLM judging.
  It requires explicit model paths through `QWEN3_CAPTION_MODEL_PATH` and
  `QWEN3_INSTRUCT_MODEL_PATH`.

- `evaluation/service_orchestrator.sh`
  Helper sourced by `run_eval.sh` for starting/stopping caption and instruct
  services and deriving service URLs.

## Core Functional Modules

### `core_functional_modules/captioner/`

- `qwen3_caption_server.py`
  Batch client for audio caption extraction through a Qwen3-Omni vLLM endpoint.

- `emotion_style_summary.py`
  Batch client that summarizes emotion/style from captions using a Qwen3-Instruct
  vLLM endpoint.

### `core_functional_modules/extract_timestamp/`

- `process_qwen3_asr.py`
  Qwen3-ASR plus forced-alignment worker. It reads packed hypothesis Parquet
  files and writes timestamp JSONL outputs. It requires `ASR_MODEL_PATH` and
  `ALIGNER_MODEL_PATH` or the matching CLI arguments.

- `combine_time_event.py`
  Combines word timestamps with detected non-verbal events and inserts event
  tags into hypothesis text.

### `core_functional_modules/PretrainedSED/`

BEATs/PretrainedSED sound-event detection wrapper used for non-verbal event
extraction.

- `batch_inference.py` runs batched SED over hypothesis Parquet files.
- `config.py` defines the checkpoint cache directory. Default:
  `~/.cache/steb/pretrained_sed`.
- `README.md` documents the local SED wrapper and upstream license note.

### `core_functional_modules/utils/`

- `parquet_io.py` provides robust Parquet/JSONL reading and writing helpers.
- `file_task_manager.py` provides simple file-level task/lock management for
  multi-process extraction.
- `vllm_client.py` provides OpenAI-compatible HTTP client utilities.
- `qwen_asr_vllm_compat.py` contains compatibility patches for Qwen3-ASR/vLLM.

## Evaluation Package

### Data Preparation

- `evaluation/eval/data_loader.py`
  Joins benchmark JSONL rows and result JSONL rows into evaluation records.

- `evaluation/eval/prepare_hyp_parquet.py`
  Packs hypothesis WAV files into Parquet for downstream feature extraction.

- `evaluation/eval/merge_hyp_features.py`
  Merges ASR, caption, summary, and event outputs into evaluation records.

- `evaluation/eval/shard_utils.py`
  Utilities for rank/world-size sharding during metric group scoring.

### Scorers

Scorers live under `evaluation/eval/scorers/`.

- `base.py` defines `EvalRecord` and scorer interfaces.
- `bleu_scorer.py` computes BLEU.
- `duration_scorer.py` and `slc_scorer.py` compute timing/SLC metrics.
- `llm_emotion_scorer.py`, `llm_style_scorer.py`, and `llm_event_scorer.py`
  run LLM-based paralinguistic judges.
- `exp_ensemble_scorer.py` repeats stochastic scorers and aggregates scores.
- `comet_scorer.py` supports optional COMET/XCOMET scoring.
- `speaker_sim_scorer.py` supports optional UniSpeech speaker similarity.
- `text_normalization.py` centralizes text cleanup for scoring.

### Vendored Speaker Similarity Code

`evaluation/eval/sim/thirdparty_unispeech/` contains vendored UniSpeech speaker
verification code. Treat it as third-party code and avoid stylistic rewrites
unless needed for a functional fix.

## Generated Outputs

`evaluation/run_eval.sh` writes outputs under `OUTPUT_DIR`, commonly:

- `eval_records.jsonl`
- `eval_records_merged.jsonl`
- `eval_results_<model>.jsonl`
- `eval_summary_<model>.json`
- logs under `logs/`
- intermediate feature directories such as `hyp_parquet/`, `hyp_timestamp/`,
  `hyp_caption/`, `hyp_summary/`, `hyp_sed/`, and `hyp_events/`

Do not commit generated outputs, virtual environments, logs, or model caches.

