# BEATs Sound Event Detection

This directory contains the BEATs-based sound-event detector used by the STEB
automatic evaluation pipeline.

The public STEB eval path calls:

```bash
python core_functional_modules/PretrainedSED/batch_inference.py \
  --parquet_path <hyp_parquet_dir> \
  --output_jsonl <hyp_sed_dir> \
  --model_name BEATs \
  --cuda
```

Only the BEATs strong checkpoint path is included in this release. Set
`PRETRAINED_SED_RESOURCES` if the checkpoint directory is not
`/home/tione/public/models/Pretrained_SED/`.

The retained BEATs implementation is derived from PretrainedSED. See
`LICENSE` in this directory for the upstream license.
