# 3. Multi-GPU and Multi-Node Training

## When to scale

- **Multi-GPU (single node)**: your model fits on one GPU but training is too slow — use DDP to parallelise across all 8 GCDs on a node.
- **Multi-node**: your dataset or model requires more memory or compute than one node can provide — extend DDP or switch to DeepSpeed.

## Launch method: `srun` vs `torchrun`

Two launchers are available. **`srun` is recommended on LUMI** because it enables CPU-GPU binding — pinning each rank to the CPU cores physically closest to its GPU, which improves memory bandwidth utilisation.

| | `srun` | `torchrun` |
|---|---|---|
| CPU-GPU binding | ✓ via `--cpu-bind` | requires code changes |
| Rank management | Slurm-native (`$SLURM_PROCID`) | handled by torchrun |
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

## RCCL environment variables

All job scripts set these — required for correct inter-node communication on LUMI:

```bash
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3  # use Slingshot interconnect
export NCCL_NET_GDR_LEVEL=PHB                   # enable GPU Direct RDMA
```

Without `NCCL_SOCKET_IFNAME`, RCCL will fail to find the correct network interface and inter-node communication will not work.

## Troubleshooting

- **Hangs at startup**: check that `MASTER_ADDR` and `MASTER_PORT` are set and reachable from all nodes
- **Slow inter-node performance**: confirm the RCCL variables above are set
- **Out of memory**: try DeepSpeed ZeRO stage 2 or 3 in `ds_config.json`

For more on distributed training on LUMI, see the [LUMI AI guide](https://docs.lumi-supercomputer.eu/software/ai/).

## Next

[4. Monitoring and Profiling](../4-monitoring-and-profiling/README.md)
