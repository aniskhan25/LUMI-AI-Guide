# 05. Synthetic Data and Data-Centric Improvement

## Goal

Use measured weak cases to generate targeted synthetic examples, filter them, and decide whether the augmented dataset actually improves the downstream task.

By the end of this lesson, you should be able to:

- explain when synthetic data is a reasonable fix and when it is not
- run a targeted synthetic-data improvement loop on LUMI
- validate that generated records, filters, provenance, and comparison artifacts stay aligned
- judge whether the augmented dataset is acceptable, risky, or inconclusive

The practical question in this lesson is:

Should I use synthetic data to address a measured weakness, or is the real problem somewhere else?

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You already know how to run Python and submit a batch job on LUMI.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/05-synthetic-data-and-data-centric-workflows
```

## What synthetic data means here

This lesson uses synthetic data in a narrow sense:

- start from measured weak cases
- generate targeted candidates
- filter them with explicit checks
- merge only accepted records
- judge success by downstream effect, not volume

This is not bulk data generation for its own sake.

## Synthetic data vs other fixes

- retrieval or prompting fixes:
  use these first if the root cause is evidence selection or answer formatting
- more real labeled data:
  preferable when authoritative labels are available and practical to collect
- model adaptation:
  useful when the model itself needs to learn the task more directly
- targeted synthetic data:
  useful when real coverage is sparse, the gap is understood, and generated examples can be validated

Use this lesson rule:

Do not generate synthetic data when the real problem is infrastructure, retrieval, or evaluation design.

## Why this baseline looks this way

The lesson uses:

- a small weak-case set
- a small baseline dataset
- template-guided candidate generation
- explicit filtering and provenance retention
- a controlled before/after rerun

The main question is:

Can I add targeted synthetic records that improve the weak cases without introducing worse regressions elsewhere?

## Main quality levers

The main choices that control this loop are:

- weak-case selection:
  synthetic data is only as useful as the weaknesses it targets
- candidates per case:
  more candidates can help coverage, but also increase noise and review burden
- filtering strictness:
  weak filters let bad synthetic records into the augmented set
- provenance retention:
  accepted records should stay traceable to their source weakness
- downstream comparison:
  success is measured by impact on the weak cases, not by accepted-record count alone

## Minimal workflow

The main path has three steps:

1. run the synthetic-data job
2. validate artifacts
3. inspect the before/after comparison

Load the lesson runtime in your shell:

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../../env.sh
```

### Step 1: Submit the synthetic-data run

Command:

```bash
sbatch jobs/run_synthdata_single_node.sh
```

This batch job runs:

- weak-case selection
- candidate generation
- filtering
- augmented dataset merge
- downstream rerun
- before/after comparison

Outputs are written to:

```bash
outputs/synthdata-loop
```

### Step 2: Validate outputs

Command:

```bash
python scripts/validate_synthdata_run.py --config configs/generate.yaml
```

Expected result:

- the Slurm log shows `GPU_VISIBLE_COUNT=1` or greater when candidate generation checks GPU visibility
- `VALIDATION_OK=1`
- candidate, accepted, rejected, and augmented dataset files exist
- rerun results exist for both baseline and augmented datasets
- `comparison.json` and `comparison_report.md` exist

This is structural success. It means the synthetic-data loop ran correctly and preserved the expected artifacts.

It does not yet mean augmentation helped.

### Step 3: Inspect the comparison

Start with:

- `outputs/synthdata-loop/comparison.json`
- `outputs/synthdata-loop/comparison_report.md`

Then inspect:

- `outputs/synthdata-loop/accepted_candidates.jsonl`
- `outputs/synthdata-loop/rejected_candidates.jsonl`

## How to read the comparison

A stronger augmentation result looks like:

- average weak-case score improves
- weak-case coverage improves
- the accepted records are clearly tied to the intended gaps
- the most improved cases match the intended weaknesses

A weaker result looks like:

- accepted volume increases but the rerun barely changes
- coverage improves a little while answer quality regresses
- accepted records look generic rather than targeted

Use this lesson rule:

If synthetic data volume goes up but downstream quality does not, the problem is selection or filtering, not generation scale.

## What this successful baseline demonstrates

If the lesson works end to end, you have shown that:

- weak cases can be turned into targeted synthetic candidates
- accepted records remain traceable to source weaknesses
- the augmented dataset stays separable from the original data
- baseline and augmented datasets can be compared through a repeatable rerun

That is different from saying synthetic data is always the right fix. The lesson teaches how to test the idea, not how to assume it will work.

## How to diagnose a bad synthetic-data loop

When the comparison is weak, ask these questions in order:

1. Were the weak cases real and relevant?
2. Did generation produce examples aligned with those weak cases?
3. Did filtering reject enough bad candidates?
4. Did the augmented dataset preserve provenance and avoid contamination?
5. Did the downstream rerun improve the intended cases?

In practice:

- many candidates, little improvement:
  revisit weak-case selection and filtering before generating more
- many rejected candidates:
  generation may be off-target, or filters may be too strict
- accepted candidates but weak rerun effect:
  inspect whether the accepted records are actually close to the measured gaps
- regressions elsewhere:
  prefer the baseline until the augmentation logic is better controlled

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Change weak-case selection before changing generation volume.
2. Adjust filtering strictness before accepting more candidates.
3. Increase `generation.num_candidates_per_case` only if the current candidates are high quality.
4. Replace the checked-in weak cases with Lesson 04 failure samples once that evaluation loop is stable.

## Troubleshooting

- missing weak cases: confirm the checked-in `data/weak_cases.jsonl` exists or that the Lesson 04 failure-sample path is valid
- `VALIDATION_OK=1` is missing: inspect candidate counts, accepted counts, and provenance fields before reading the comparison
- poor augmentation result: inspect accepted and rejected examples before generating more data

## Next lesson

Next extension lesson: topology-aware scaling of advanced AI workloads.
