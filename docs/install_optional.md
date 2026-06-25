# Optional Environment Installation

## Optional COMET / XCOMET scoring

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

## Optional speaker similarity scoring

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