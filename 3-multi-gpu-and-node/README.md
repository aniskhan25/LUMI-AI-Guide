# 3. Multi-GPU and Multi-Node Training

## LUMI-G node topology

Understanding the hardware layout helps you set up jobs correctly and interpret scaling results.

A full LUMI-G node exposes:

- **8 GPU-visible devices (GCDs)** — 4 AMD MI250X modules, each with 2 Graphic Compute Dies
- **56 CPU cores** — 1 AMD EPYC Trento, 4 NUMA domains
- **4 Slingshot NICs** — one per MI250X module, used for inter-node communication

Two facts matter immediately for scaling:

1. The two GCDs inside one MI250X communicate over a fast in-package Infinity Fabric link. GCDs across different MI250X modules use slower cross-package links.
2. Each scaling step changes the communication pattern: `1 GCD → 8 GCDs (one node) → multi-node` each add a new communication layer, not just more devices.

This is why CPU-GPU binding matters — each rank should be pinned to the CPU cores in the same NUMA domain as its GPU. The `srun` scripts in this lesson handle this with `--cpu-bind=mask_cpu`.

For more detail see the [LUMI-G hardware overview](https://docs.lumi-supercomputer.eu/hardware/lumig/) and [distribution and binding guide](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/).

## When to scale

- **Multi-GPU (single node)**: your model fits on one GPU but training is too slow — use DDP to parallelise across all 8 GCDs on a node.
- **Multi-node**: your dataset or model requires more memory or compute than one node can provide — extend DDP or switch to DeepSpeed.
- **Don't scale yet**: if your single-GPU utilisation is low or your data pipeline is the bottleneck, fix that first — more GPUs won't help.

## GPU-hour planning

LUMI bills by GPU-hours consumed. Before scaling up, use a staged approach:

| Stage | Purpose | Partition |
|---|---|---|
| Debug (1 GCD) | Confirm the script runs end-to-end | `dev-g` |
| Baseline (1 node, 8 GCDs) | Measure single-node throughput | `standard-g` |
| Scale test (multi-node) | Check whether more nodes improve throughput enough | `standard-g` |

Move to the next stage only after the current one answers its question. Jumping straight to multi-node before a stable single-node baseline is the most common way to waste GPU-hours on LUMI.

## Launch method: `srun` vs `torchrun`

Two launchers are available. **`srun` is recommended on LUMI** because it enables CPU-GPU binding — pinning each rank to the CPU cores physically closest to its GPU, which improves memory bandwidth utilisation.

| | `srun` | `torchrun` |
|---|---|---|
| CPU-GPU binding | ✓ via `--cpu-bind=mask_cpu` | ✓ via `--numa-binding=exclusive` |
| Rank management | SLURM-native (`$SLURM_PROCID`) | handled by torchrun |
| Recommended on LUMI | ✓ | |

## PyTorch DDP

Use DDP when your model fits on a single GPU. Key changes in `visiontransformer_ddp.py`:

```python
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = DistributedDataParallel(model, device_ids=[local_rank])

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sampler, ...)
```

### Single node (8 GPUs)

```bash
sbatch run_ddp_srun.sh        # recommended
sbatch run_ddp_torchrun.sh    # alternative
```

### Multi-node (4 nodes, 32 GPUs)

```bash
sbatch run_ddp_srun_n4.sh
sbatch run_ddp_torchrun_n4.sh
```

## DeepSpeed

Use DeepSpeed when your model does not fit on a single GPU. It shards model parameters across GPUs and optionally offloads to CPU memory. Configure the sharding strategy in `ds_config.json` — see the [DeepSpeed ZeRO documentation](https://www.deepspeed.ai/tutorials/zero/) for options.

### Single node

```bash
sbatch run_deepspeed_srun.sh
sbatch run_deepspeed_torchrun.sh
```

### Multi-node (4 nodes)

```bash
sbatch run_deepspeed_srun_n4.sh
sbatch run_deepspeed_torchrun_n4.sh
```

## Is scaling actually helping?

Before requesting more nodes, measure whether your workload benefits from the GPUs you already have. [Scaling-Aware AI on LUMI](https://github.com/aniskhan25/scaling-aware-ai) walks through a structured approach — baseline experiments, bottleneck diagnosis (data starvation, load imbalance, communication overhead), and a clear decision framework for when multi-node is justified.

## Troubleshooting

- **Hangs at startup**: check that `MASTER_ADDR` and `MASTER_PORT` are set and reachable from all nodes
- **Out of memory**: try DeepSpeed ZeRO stage 2 or 3 in `ds_config.json`

For more on distributed training on LUMI, see the [LUMI AI guide](https://docs.lumi-supercomputer.eu/software/ai/).

## Next

[4. Monitoring and Profiling](../4-monitoring-and-profiling/README.md)
