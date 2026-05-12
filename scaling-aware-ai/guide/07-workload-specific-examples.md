# 7. Workload-Specific Examples

Pass 3 adds two representative workload patterns.

## Example A: Synthetic DDP Training

Files:

- `jobs/run_ddp_1gcd.sh`
- `jobs/run_ddp_8gcd_single_node.sh`
- `scripts/run_ddp_training.py`
- `configs/ddp-training/baseline.yaml`
- `configs/ddp-training/single_node.yaml`
- `examples/ddp-training/README.md`

Use this when the workload has synchronized model updates.

Run:

```bash
sbatch jobs/run_ddp_1gcd.sh
sbatch jobs/run_ddp_8gcd_single_node.sh
```

Compare:

- total throughput
- mean rank throughput
- min/max rank throughput
- data wait fraction
- checkpoint time
- placement files

Interpretation:

- poor 1-GCD throughput means the baseline needs work
- poor 8-GCD throughput with low data wait suggests communication or launch overhead
- poor 8-GCD throughput with high data wait suggests input delivery is the bottleneck
- high checkpoint time means wall-clock speedup may not translate into end-to-end speedup

## Example B: Batch Inference Job Array

Files:

- `jobs/run_batch_inference_array.sh`
- `scripts/run_batch_inference.py`
- `scripts/collect_batch_inference.py`
- `configs/batch-inference/job_array.yaml`
- `examples/batch-inference/README.md`
- `examples/batch-inference/data/sample_requests.jsonl`

Use this when records are independent.

Run:

```bash
sbatch jobs/run_batch_inference_array.sh
```

After the array finishes:

```bash
python scripts/collect_batch_inference.py --config configs/batch-inference/job_array.yaml
```

Compare:

- records written
- shards completed
- max shard elapsed time
- records/sec by max shard elapsed time
- missing shard summaries

Interpretation:

- if shards are independent and balanced, job arrays avoid distributed communication entirely
- if max shard elapsed time is much larger than the rest, improve sharding
- if records are missing, fix restart and aggregation before reporting throughput

## Choosing Between The Examples

Use DDP when:

- ranks share one training job
- gradients synchronize
- global batch size matters
- every step depends on all ranks

Use job arrays when:

- records are independent
- outputs can be merged later
- retry per shard is useful
- throughput matters more than collective synchronization

## Practical Rule

Distributed frameworks solve synchronization problems. Job arrays solve independent-work throughput problems.

