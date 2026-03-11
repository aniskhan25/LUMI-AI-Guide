# Common Failures

## 1) No GPU visible inside container

Symptoms:

- `GPU_VISIBLE_COUNT=0`
- train script exits with CUDA visibility error

Checks:

- Ensure binding module is loaded in job script:
  - `module load lumi-aif-singularity-bindings` (or fallback)
- Confirm job runs on a GPU partition (`small-g` or `standard-g`)

## 2) Missing or wrong container path

Symptoms:

- `Set CONTAINER in env.sh`
- singularity exec fails before python starts

Checks:

- Set valid `CONTAINER` in [env.sh](../../../env.sh)
- Verify path exists on LUMI and is readable

## 3) JSONL parse or key errors

Symptoms:

- `KeyError: text` or `KeyError: label`
- JSON decode failures

Checks:

- Rebuild sample data:
  - `python data/prepare_sample_data.py --output data/sample_data`
- Match keys in `configs/baseline.yaml` (`text_key`, `label_key`)

## 4) Out-of-memory

Symptoms:

- Runtime OOM
- sudden process termination during forward pass

Checks:

- Reduce `training.batch_size`
- Reduce `data.max_seq_len`
- Use `adaptation.mode=head_only` as baseline

## 5) Poor scaling assumptions

Symptoms:

- Full-node runs are slower than expected

Checks:

- Do not assume simple CPU/GPU numbering alignment on MI250X/GCD topology
- Start from single-device baseline and profile before scaling decisions

