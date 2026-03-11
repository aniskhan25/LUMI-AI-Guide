# Common Failures

## 1) Accidental CPU execution

Symptoms:

- `gpu_visible_count` is `0` in summary
- run is much slower than expected

Fix:

- load container bindings module in job script
- submit to a GPU partition
- require GPU in validation (`--require-gpu`)

## 2) Output count mismatch

Symptoms:

- validation fails with input/output count mismatch
- some IDs missing in output

Fix:

- confirm job did not terminate early
- verify no filtering logic dropped records unintentionally
- rerun and validate IDs with `scripts/validate_outputs.py`

## 3) Embedding dimension mismatch

Symptoms:

- validation reports inconsistent embedding dimensions

Fix:

- ensure a single model checkpoint is used for the whole run
- avoid mixed output files from different runs in the same path
- clear output directory before rerun

## 4) Out-of-memory during inference

Symptoms:

- runtime OOM or abrupt process termination

Fix:

- lower `inference.batch_size`
- lower sequence length (`max_seq_len` or `max_input_length`)
- use a smaller model

## 5) Ordering assumptions break downstream mapping

Symptoms:

- downstream process expects strict input/output ordering

Fix:

- preserve and use `id` keys as the contract
- do not rely on record order alone
- validate ID set equality before downstream use

