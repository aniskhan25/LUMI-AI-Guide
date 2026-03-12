# Expected Schemas

## Per-rank placement record (`raw/placement_rank*.json`)

Required fields:

- `rank`
- `local_rank`
- `world_size`
- `hostname`
- `cuda_visible_devices`
- `slurm_localid`
- `slurm_procid`
- `cpus_per_task`

## Per-rank metrics record (`raw/metrics_rank*.json`)

Required fields:

- `rank`
- `world_size`
- `device`
- `steps`
- `samples_per_step`
- `elapsed_seconds`
- `throughput_samples_per_sec`

## Run summary (`run_summary.json`)

Required fields:

- `run_name`
- `world_size`
- `node_count`
- `gpu_visible_count`
- `mean_throughput_samples_per_sec`
- `max_elapsed_seconds`
- `effective_samples_per_step`

## Scaling report (`scaling_report.json`)

Required fields:

- baseline/single-node/multi-node throughput
- speedup vs baseline
- efficiency estimates
- diagnosis notes

