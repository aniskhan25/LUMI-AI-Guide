# 04. Evaluation and Trustworthiness

## Goal

Evaluate two controlled workflow variants on the same dataset and produce evidence for a deployment decision.

By the end of this lesson, you should be able to:

- explain what evaluation, comparison, and trustworthiness mean in this guide
- run a repeatable evaluation job on LUMI
- inspect both aggregate metrics and concrete failure cases
- justify a baseline-versus-candidate recommendation from saved artifacts

## Assumptions

- You completed [1. QuickStart](../../1-quickstart/README.md).
- You completed [2. Setting up your own environment](../../2-setting-up-environment/README.md).
- You already know how to run Python and submit a batch job on LUMI.
- `../../env.sh` is configured with a valid `CONTAINER`.

## Working directory

Run commands in this lesson from:

```bash
cd /path/to/LUMI-AI-Guide/extension-track/04-evaluation-and-trustworthiness
```

## What evaluation means here

This lesson uses three ideas together:

- evaluation: measure whether outputs meet the intended task
- comparison: test one controlled change against a baseline
- trustworthiness: keep enough evidence to explain the decision later

Here, the controlled change is retrieval depth:

- baseline: `top_k=3`
- candidate: `top_k=5`

The evaluation set is small and curated on purpose. It gives each query:

- an expected source document
- a reference answer
- required terms for lightweight scoring

That makes the results inspectable instead of purely aggregate.

## Minimal workflow

The main path has three steps:

1. run the evaluation job
2. validate the artifacts
3. read the comparison and report

Load the lesson runtime in your shell:

```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
source ../../env.sh
```

### Step 1: Submit the evaluation run

Command:

```bash
sbatch jobs/run_eval_single_node.sh
```

This batch job runs:

- baseline variant
- candidate variant
- scoring
- failure extraction
- comparison
- report generation

Outputs are written to:

```bash
outputs/eval-rag
```

### Step 2: Validate outputs

Command:

```bash
python scripts/validate_eval_run.py --config configs/eval.yaml
```

Expected result:

- the Slurm log shows `GPU_VISIBLE_COUNT=1` or greater for the workflow runs
- `VALIDATION_OK=1`
- both variants have scored outputs for every evaluation item
- `comparison.json` exists
- `evaluation_report.md` exists

### Step 3: Read the decision artifacts

Start with:

- `outputs/eval-rag/comparison.json`
- `outputs/eval-rag/evaluation_report.md`

Then inspect:

- `outputs/eval-rag/baseline/failure_samples.jsonl`
- `outputs/eval-rag/candidate/failure_samples.jsonl`

## What the metrics mean

This lesson uses a small operational scorecard:

- `retrieval_hit_rate`: did the answer cite chunks from the expected document
- `answer_score_mean`: how many required reference terms appeared
- `grounded_rate`: did retrieval and answer content align well enough to count as grounded
- `completion_rate`: did the system return a non-empty answer

These are comparison metrics, not universal truth. Their job is to help you compare variants on the same evaluation set.

## Why failure samples matter

Aggregate metrics tell you whether something changed.

Failure samples tell you why.

In this lesson, a recommendation is only credible if both agree:

- the candidate improves or preserves the important metrics
- the remaining failures are understandable and acceptable

## What this successful baseline demonstrates

If the lesson works end to end, you have shown that:

- both variants were evaluated on the same dataset
- outputs, scores, and failures remain tied to stable query IDs
- the comparison is reproducible
- the recommendation is backed by saved artifacts rather than intuition

## What to change next

After the first successful run, change one thing at a time.

Recommended order:

1. Change only one retrieval parameter, such as `top_k`.
2. Review whether metric changes match failure-sample changes.
3. Replace the evaluation set with your own queries and references.
4. Add stricter scoring only after the basic workflow is stable.

## Troubleshooting

- missing outputs for one variant: check that both variant directories exist before debugging scoring
- `VALIDATION_OK=1` is missing: inspect count mismatches and missing files before interpreting metrics
- weak recommendation quality: inspect `failure_samples.jsonl` before changing the metric weights

## Next lesson

Next extension lesson: synthetic data and data-centric workflows.
