# 06. Topology-Aware Scaling

## Goal

Run a small scaling ladder on LUMI and decide whether adding more GPUs or nodes actually improves useful throughput.

By the end of this lesson, you should be able to:

- explain when scaling is worth trying and when it is not
- run a 1-GCD, 8-GCD, and small multi-node comparison
- validate that rank counts, placement metadata, and summaries align
- judge whether a scaling result is acceptable, moderate, or poor

The practical question in this lesson is:

When is it worth scaling beyond one GPU-visible device, and how do I know if topology is helping or hurting?

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

## What scaling means here

This lesson treats scaling as a mapping experiment:

- more devices should increase useful throughput
- rank placement should match the node layout
- communication overhead should stay small enough to justify the extra devices

This is not a general distributed-systems tutorial. It is a practical lesson about reading a scaling result on LUMI.

## Scaling vs other fixes

- baseline optimization:
  use this first if one-device throughput is still poor
- data or storage optimization:
  use this if input delivery is the real bottleneck
- topology-aware scaling:
  use this when the single-device workload is already stable and the job is large enough to benefit from more devices

Use this lesson rule:

Do not scale a workload that is still unstable or too small to justify communication overhead.

## Why this baseline looks this way

The lesson compares:

- 1 GCD on 1 node
- 8 GCDs on 1 node
- 16 GCDs on 2 nodes

The workload stays effectively the same across runs so the comparison stays meaningful.

The main question is:

Did extra devices improve useful throughput enough to justify the added communication and placement complexity?

## Main quality levers

The main choices that control scaling behavior in this lesson are:

- world size:
  more ranks increase compute and communication together
- node count:
  multi-node runs add network communication, not just more devices
- rank placement:
  GPU-visible device order and CPU binding affect scaling outcomes
- effective workload size:
  tiny workloads often scale badly because communication dominates

## Minimal workflow

The main path has three steps:

1. run the scaling ladder
2. validate the summaries
3. inspect the scaling report

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

These produce:

- `outputs/scaling-1gcd`
- `outputs/scaling-8gcd-single-node`
- `outputs/scaling-multi-node`

Then build the comparison:

```bash
python scripts/compare_scaling.py
```

### Step 2: Validate outputs

Command:

```bash
python scripts/validate_scaling_run.py
```

Expected result:

- each run directory has placement and metrics records
- each run has `run_summary.json`
- `scaling_report.json` and `scaling_report.md` exist
- `VALIDATION_OK=1`

This is structural success. It means the scaling ladder ran correctly and produced comparable summaries.

It does not yet mean scaling was worthwhile.

### Step 3: Inspect the scaling report

Start with:

- `outputs/scaling_report.json`
- `outputs/scaling_report.md`

The main fields to read are:

- total throughput
- speedup vs baseline
- efficiency vs baseline
- diagnosis

## How to read the scaling result

A stronger scaling result looks like:

- throughput rises clearly as world size grows
- speedup is meaningful relative to the extra devices
- efficiency stays reasonably high
- the diagnosis stays favorable

A weaker scaling result looks like:

- throughput rises only slightly while device count rises a lot
- efficiency collapses as communication increases
- multi-node performance is much worse than single-node scaling

Use this lesson rule:

More GPUs are only useful if they improve useful throughput enough to justify the extra communication.

## How to diagnose poor scaling

When scaling looks weak, ask these questions in order:

1. Is the effective workload large enough?
2. Did the expected world size and node count actually match the run?
3. Does placement metadata look plausible for the launched ranks?
4. Is the slowdown mostly appearing at single-node scale or only at multi-node scale?

In practice:

- weak 8-GCD scaling:
  inspect placement and workload size before trying multi-node
- good 8-GCD scaling but weak 2-node scaling:
  communication overhead is likely dominating across nodes
- mismatched rank counts or missing placement files:
  fix the launch first before interpreting throughput

## What this successful baseline demonstrates

If the lesson works end to end, you have shown that:

- the workload can be launched consistently across 1-device, 1-node, and multi-node settings
- placement metadata and throughput summaries remain comparable
- speedup and efficiency can be interpreted together
- topology is part of the result, not just background detail

That is different from saying “more GPUs always help.” The lesson teaches how to test the scaling decision, not how to assume it.

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Increase workload size before changing placement assumptions.
2. Compare 1 GCD vs 8 GCD carefully before moving to multi-node.
3. Revisit CPU binding or distribution only after the baseline ladder is understood.
4. Extend to a larger communication-heavy workload only after the summary metrics make sense.

## Troubleshooting

- missing rank or placement files: fix the launch before reading throughput
- poor efficiency with tiny workloads: increase the workload before blaming topology
- multi-node regression after good single-node scaling: inspect cross-node communication assumptions first

## Next lesson

Next extension lesson: advanced inference and serving patterns.
