# 06. Topology-Aware Scaling

## Goal

Run a small scaling ladder on LUMI and decide whether adding more GPUs or nodes actually improves useful throughput.

By the end of this lesson, you should be able to:

- explain what a LUMI-G node exposes to software and why that matters for scaling
- run a 1-GCD, 8-GCD, and 16-GCD comparison
- validate that rank counts, placement metadata, and summaries align
- read speedup and efficiency as topology signals, not just bigger-is-better numbers

The practical question in this lesson is:

When is it worth scaling beyond one GPU-visible device, and how do I know if LUMI-G topology is helping or hurting?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You already know how to run Python and submit a batch job on LUMI.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/06-topology-aware-scaling
```

## Why this lesson needs more care

Scaling is one of the easiest AI workflow topics to misread.

If a larger run is slower than expected, the problem may be:

- a workload that is too small
- weak CPU or GPU placement
- intra-node communication cost
- inter-node communication cost
- a launch mismatch that invalidates the comparison

So this lesson is not just "run three jobs and compare numbers." It is a controlled experiment about how a workload maps onto LUMI-G.

## What a LUMI-G node actually is

This lesson uses the software view of a LUMI-G node, because that is what Slurm and PyTorch expose:

- 4 AMD MI250X modules per node
- 2 GCDs per MI250X
- 8 GPU-visible devices per full node
- 1 AMD EPYC Trento CPU with 64 physical cores, of which 56 are available to jobs on LUMI-G
- 4 CPU NUMA domains
- 4 Slingshot network endpoints, one per MI250X module

Two topology facts matter immediately:

1. A full LUMI-G node is an 8-GCD node, not a 4-GPU node from the point of view of Slurm and HIP.
2. NUMA numbering does not line up directly with GPU numbering, so rank placement and CPU affinity can affect observed performance.

The LUMI documentation also notes that:

- the two GCDs inside one MI250X communicate over an in-package Infinity Fabric link
- GCDs on different MI250X modules communicate over single or double Infinity Fabric links
- multi-node communication adds Slingshot network behavior on top of GPU-to-GPU communication

That is why `1 GCD -> 8 GCDs on one node -> 16 GCDs on two nodes` is a meaningful scaling ladder. Each step changes the communication pattern, not just the device count.

## What this lesson is actually measuring

The workload in this lesson is intentionally synthetic. It is not trying to benchmark a real model. It is trying to expose scaling behavior in a controlled way.

The script [run_workload.py](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/06-topology-aware-scaling/scripts/run_workload.py) does three important things:

- runs repeated dense GPU compute
- keeps the workload shape fixed across the ladder
- performs an `all_reduce` in distributed runs so communication shows up in the result

The config values are the same in all three runs:

- `samples_per_step: 512`
- `steps: 40`
- `warmup_steps: 5`
- `hidden_size: 2048`
- `compute_repeats: 6`

That makes the comparison controlled:

- the single-GCD run measures the baseline throughput of one visible device
- the 8-GCD run measures how well the same workload scales within one node
- the 16-GCD run measures what changes once network communication is introduced

This is enough to teach scaling interpretation without pretending to be a model-quality benchmark.

## When scaling is worth trying

Scale only after the single-device case is already stable and worth speeding up.

Scaling is usually the right next move when:

- one visible device is already well utilized
- the workload is large enough that compute dominates launch overhead
- the job is repeated often enough that faster turnaround matters
- you expect more devices to reduce wall-clock time enough to justify the extra cost and complexity

Scaling is usually the wrong next move when:

- one-device throughput is still poor for local reasons
- data loading or storage is the real bottleneck
- the workload is too small to amortize communication
- the launch pattern itself is not yet trustworthy

Use this lesson rule:

Do not scale an unstable or undersized workload.

## Scaling vs other fixes

- baseline optimization:
  use this first if one-device throughput is still poor
- data or storage optimization:
  use this if input delivery is the bottleneck
- topology-aware scaling:
  use this when the single-device run is already healthy and the remaining question is whether more devices help

This distinction matters because topology cannot rescue a weak single-device baseline.

## The scaling ladder in this lesson

The lesson uses three launch patterns:

1. `jobs/run_1gcd.sh`
   one task, one visible device, `dev-g`
2. `jobs/run_8gcd_single_node.sh`
   one full LUMI-G node, `torch.distributed.run`, world size `8`
3. `jobs/run_multi_node.sh`
   two full LUMI-G nodes, `torch.distributed.run`, world size `16`

The expected outputs are:

- `outputs/scaling-1gcd`
- `outputs/scaling-8gcd-single-node`
- `outputs/scaling-multi-node`
- `outputs/scaling_report.json`
- `outputs/scaling_report.md`

The first job gives the baseline. The second adds only intra-node communication. The third adds inter-node communication as well.

The launchers are intentionally simple. They are good enough to demonstrate scaling behavior and collect placement metadata, but they are not presented as the final word on LUMI-G binding or communication tuning.

## Main quality levers

The choices that matter most in this lesson are:

- world size:
  more ranks add compute and communication together
- node count:
  multi-node runs introduce network effects, not just more devices
- placement:
  CPU affinity, rank distribution, and GPU visibility affect locality
- workload size:
  a tiny workload often scales badly because overhead dominates
- communication pattern:
  collective operations can make a larger run slower than expected even when compute scales well

These are the levers you should reason about before changing model code.

## What the scripts record

The lesson records two kinds of per-rank data:

- placement metadata from [inspect_placement.py](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/06-topology-aware-scaling/scripts/inspect_placement.py)
- throughput metrics from [run_workload.py](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/06-topology-aware-scaling/scripts/run_workload.py)

Those are aggregated by [collect_metrics.py](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/06-topology-aware-scaling/scripts/collect_metrics.py) into `run_summary.json`.

The final comparison is built by [compare_scaling.py](/Users/anisrahm/Documents/LUMI-AI-Guide/extension-track/06-topology-aware-scaling/scripts/compare_scaling.py), which reports:

- total throughput
- speedup versus baseline
- efficiency versus baseline
- a short diagnosis

## How the lesson computes the main numbers

The report is small, but the meanings matter:

- `total throughput`:
  the sum of per-rank throughput across the run
- `speedup vs baseline`:
  `throughput_large / throughput_1gcd`
- `efficiency vs baseline`:
  `speedup / (world_size_large / world_size_baseline)`

Examples:

- perfect `8x` speedup from `1` to `8` ranks would give efficiency `1.0`
- `4x` speedup on `8` ranks gives efficiency `0.5`
- `12x` speedup on `16` ranks gives efficiency `0.75`

Efficiency matters because raw throughput almost always rises when you add enough devices. The real question is whether it rises enough.

## Minimal workflow

The workflow stays short even though the explanation is longer.

Load the lesson runtime in your shell:

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../../env.sh
```

### Step 1: Submit the scaling runs

Commands:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
sbatch jobs/run_multi_node.sh
```

What each script does:

- `run_1gcd.sh`:
  captures placement, runs the workload once on one visible device, builds a one-rank summary
- `run_8gcd_single_node.sh`:
  launches eight ranks on one node with `torch.distributed.run`, then builds a summary
- `run_multi_node.sh`:
  launches sixteen ranks across two nodes, sets rendezvous information, then builds a summary

After all three runs finish, build the comparison:

```bash
python scripts/compare_scaling.py
```

### Step 2: Validate outputs

Command:

```bash
python scripts/validate_scaling_run.py
```

Expected result:

- each run directory has placement and metrics files
- each run has `run_summary.json`
- world size and node count match the config
- `scaling_report.json` and `scaling_report.md` exist
- `VALIDATION_OK=1`

This is structural success. It means the ladder ran correctly and produced comparable summaries.

It does not yet mean scaling was worthwhile.

### Step 3: Inspect the report and the raw placement data

Start with:

- `outputs/scaling_report.json`
- `outputs/scaling_report.md`

Then inspect one raw placement file from each run, for example:

- `outputs/scaling-1gcd/raw/placement_rank0.json`
- `outputs/scaling-8gcd-single-node/raw/placement_rank0.json`
- `outputs/scaling-multi-node/raw/placement_rank0.json`

The raw placement files tell you:

- hostname
- rank and local rank
- visible GPU count
- `CUDA_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` if set
- Slurm job metadata

That is the first place to look if the comparison smells wrong.

## How to read the scaling result

A stronger result looks like this:

- 8 GCDs materially outperform 1 GCD
- 16 GCDs materially outperform 8 GCDs
- efficiency drops somewhat, but not catastrophically
- the diagnosis remains favorable or at least moderate

A weaker result looks like this:

- throughput rises only modestly while world size rises a lot
- 8-GCD efficiency is already poor
- the 16-GCD run adds little over the 8-GCD run
- the diagnosis points to communication or mapping cost

Use this lesson rule:

More devices are only useful if they improve useful throughput enough to justify the added communication.

## What a bad result can still mean

Poor scaling is not automatically a failed lesson. It may be the correct result.

For example:

- if the workload is too small, poor efficiency is expected
- if 8-GCD scaling is decent but 16-GCD scaling drops sharply, network communication is likely the real limit
- if rank counts or placement files do not match the launch, the result is invalid and should not be interpreted

So there are three different outcomes:

- good scaling:
  larger configurations are worth using
- poor but valid scaling:
  the workload or communication pattern does not justify more devices
- invalid scaling:
  the launch did not produce a trustworthy comparison

Only the first outcome supports scaling up in production.

## How to diagnose poor scaling

When scaling looks weak, ask these questions in order:

1. Is the workload large enough to amortize collective communication?
2. Did the observed `world_size` and `node_count` match the intended launch?
3. Did the raw placement files show the expected hostnames and rank counts?
4. Did the efficiency drop already at 8 GCDs, or only when moving to 2 nodes?

Interpretation:

- weak 8-GCD scaling:
  inspect placement and workload size before blaming the network
- good 8-GCD scaling but weak 16-GCD scaling:
  inter-node communication is likely the main limit
- mismatched rank counts or missing placement files:
  fix the launch first and ignore the throughput numbers

## What this lesson does not prove

Even if the lesson runs perfectly, it does not prove:

- that your real model will scale identically
- that the current binding is optimal
- that a larger job is cost-effective
- that communication libraries are tuned as far as they can go

This lesson proves something narrower and still useful:

- the workload can be launched consistently across 1, 8, and 16 visible devices
- placement metadata and throughput summaries remain comparable
- speedup and efficiency can be interpreted together
- topology is part of the result, not just background detail

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Increase workload size before changing placement assumptions.
2. Compare 1 GCD vs 8 GCD carefully before moving to stronger conclusions about multi-node behavior.
3. Revisit CPU and GPU binding if 8-GCD scaling is weaker than expected.
4. Extend to a more communication-heavy workload only after the baseline ladder is understood.

## Troubleshooting

- missing rank or placement files:
  fix the launch before reading throughput
- poor efficiency with tiny workloads:
  increase the workload before blaming topology
- multi-node regression after acceptable single-node scaling:
  inspect inter-node communication assumptions first

## Source notes

This lesson is grounded in the official LUMI documentation and training materials:

- [LUMI-G hardware overview](https://docs.lumi-supercomputer.eu/hardware/lumig/)
- [LUMI network and interconnect](https://docs.lumi-supercomputer.eu/hardware/network/)
- [LUMI distribution and binding guide](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/)
- [LUMI-G example batch scripts](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/)
- [LUMI architecture training materials](https://lumi-supercomputer.github.io/LUMI-training-materials/2day-20240502/01_Architecture/)

These are worth reading after the lesson if you want to tune beyond this tutorial's deliberately simple launch scripts.

## Next lesson

Next extension lesson: advanced inference and serving patterns.
