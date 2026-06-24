# STEB Agent Runbook

This file summarizes the current repository cleanup and the practical commands
other code agents should use when modifying or testing STEB.

## What Was Updated

The repository has been prepared for a public academic code release:

- The public README was rewritten into a paper-project style README with
  Paper/Homepage/Hugging Face slots.
- Machine-specific defaults were removed from scripts. Dataset, output, model,
  COMET, XCOMET, ASR, and vLLM paths must now be supplied explicitly.
- Internal network proxy defaults were removed.
- Internal experiment-version names were removed from public docs and code.
- Chinese/private comments were removed except user-facing language aliases in
  ASR language normalization.
- `.gitignore` was added for Python caches, logs, virtual environments, and
  common output directories.
- `pyproject.toml` and `uv.lock` now define the main `uv` environment.
- Optional COMET/XCOMET and speaker-similarity workflows are documented as
  isolated environments because they have dependency constraints that should not
  be mixed with the default vLLM environment.

## Default Environment

Use `uv` from the repository root:

```bash
uv sync --python 3.10
uv run python -c "import torch, torchaudio, vllm, pandas, sacrebleu; print('STEB environment OK')"
```

If a platform needs a custom PyTorch wheel, create the environment and install
PyTorch first:

```bash
uv venv --python 3.10
uv pip install torch torchaudio --index-url <pytorch-wheel-index-url>
uv sync
```

`requirements.txt` remains for legacy workflows:

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

## End-to-End Evaluation

Primary command shape:

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

Use `SPLIT=event` for event-bearing samples. That enables BEATs SED and event
combination before scoring.

## Useful Switches

- `START_PHASE=3 END_PHASE=3`
  Run scoring only from existing `eval_records.jsonl` or
  `eval_records_merged.jsonl`.

- `SKIP_PHASE2_ASR=1`
  Skip ASR/timestamp extraction when outputs already exist.

- `SKIP_PHASE2_CAPTION=1`
  Skip audio caption extraction.

- `SKIP_PHASE2_SUMMARY=1`
  Skip emotion/style summarization.

- `SKIP_PHASE2_SED=1`
  Skip BEATs sound-event detection.

- `SKIP_PHASE2_EVENT_COMBINE=1`
  Skip timestamp/event merging.

- `AUTO_START_CAPTION_SERVERS=0 CAPTION_SERVER_URLS=http://host:port/v1`
  Use manually launched caption servers.

- `AUTO_START_INSTRUCT_SERVERS=0 INSTRUCT_SERVER_URLS=http://host:port/v1`
  Use manually launched instruct servers.

## Optional COMET / XCOMET

COMET is intentionally isolated from the default vLLM environment.

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

For XCOMET, pass:

```bash
--enable_xcomet --xcomet_model <model-or-path>
```

## Optional Speaker Similarity

Speaker similarity uses the vendored UniSpeech speaker-verification stack in an
isolated environment:

```bash
uv venv .envs/speaker-sim --python 3.10
uv pip install --python .envs/speaker-sim/bin/python \
  torch torchaudio tqdm soundfile librosa packaging omegaconf \
  s3prl==0.3.1 fairseq==0.12.2 fire

export SPEAKER_SIM_PYTHON=$PWD/.envs/speaker-sim/bin/python
export SPEAKER_SIM_CKPT=/path/to/wavlm_large_finetune.pth
```

Then add:

```bash
ENABLE_SPEAKER_SIM=--enable_speaker_sim
```

`evaluation/run_eval.sh` validates `SPEAKER_SIM_PYTHON` and
`SPEAKER_SIM_CKPT` before running speaker similarity.

## Data Format Reminder

Benchmark JSONL rows need source/reference fields such as:

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

Result JSONL rows need hypothesis fields such as:

```json
{
  "id": "sample_001",
  "hyp_text": "Translation with sound events: ...",
  "hyp_wav_path": "/path/to/hypothesis.wav",
  "model_name": "my_model",
  "error": null
}
```

## Verification For Code Agents

Before handing off changes, run the checks that do not require model weights or
GPU execution:

```bash
bash -n evaluation/run_eval.sh
bash -n core_functional_modules/start_servers.sh
bash -n evaluation/service_orchestrator.sh
python - <<'PY'
import ast
import pathlib
import sys

bad = []
for root in (pathlib.Path("core_functional_modules"), pathlib.Path("evaluation")):
    for path in root.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            bad.append((str(path), exc))

if bad:
    for path, exc in bad:
        print(f"{path}: {exc}")
    sys.exit(1)

print("syntax ok")
PY
git diff --check
```

Do not leave `__pycache__/` directories in the working tree.

Also re-run release hygiene scans before public release:

```bash
rg -n "(/home/[^/]+|/[a]pdcephfs|10\\.[0-9]+\\.[0-9]+\\.[0-9]+|notebook/(project|model)|public/(models|datasets))" \
  -g '!*.th' -g '!*.png' -g '!*.jpg' -g '!*.pt' -g '!*.pth' -g '!*.bin'
rg -n "v[0-9]+(_[A-Za-z0-9]+)?" README.md evaluation core_functional_modules \
  -g '!*.th' -g '!*.pt' -g '!*.pth'
```

The first scan should return no matches. The second scan may match legitimate
dependency names or APIs; inspect any result and remove internal experiment
version names from public-facing docs, logs, and option defaults.

## Editing Notes

- Keep user-facing docs free of internal paths, internal proxy addresses, and
  experiment-version names.
- Keep model/data paths explicit through environment variables or CLI flags.
- Avoid broad refactors in vendored third-party code unless needed for a bug.
- Do not commit `.venv/`, `.envs/`, logs, generated evaluation outputs, model
  checkpoints, or cache directories.
