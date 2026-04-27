# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 01.

## 1. Lesson Identity

- Lesson id: `EXT-01`
- Title: `Adapting Foundation Models on LUMI-G with the AI Factory Software Environment`
- Short nav title: `Foundation model adaptation on LUMI-G`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

Fill this table before changing lesson prose or scripts.

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Example task | Sequence classification, compact instruction tuning | `TBD` | `TBD` | `TBD` | `TBD` |
| Model family | e.g. DistilBERT, Qwen2.5-Instruct small, Llama small | `TBD` | `TBD` | `TBD` | `TBD` |
| Adaptation mode | Full fine-tune, PEFT/LoRA, head-only baseline | `TBD` | `TBD` | `TBD` | `TBD` |
| Primary validation artifact | Accuracy/F1/perplexity/loss reduction | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- One end-to-end adaptation workflow on LUMI-G
- One recommended container-first execution path
- One minimal dataset and config pattern
- One verification checklist and troubleshooting page

### Out Of Scope

- Account setup and generic Slurm primer
- Full multi-node distributed strategy
- General container theory
- RAG and serving architecture

## 4. Learning Outcomes

By lesson end, learner can:

1. define foundation-model adaptation in this context
2. run a supported adaptation job on LUMI-G
3. distinguish full fine-tuning vs parameter-efficient adaptation
4. verify GPU use and artifacts
5. perform one safe modification to the baseline run

## 5. Lesson Structure Contract

The published lesson must include these sections:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal working example
- F. Verification
- G. Why this works on LUMI-G
- H. Common failure modes
- I. Extension paths
- J. Operational checklist
- K. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Minimal training entrypoint | Yes | `scripts/train.py` |
| Config file | Yes | `configs/baseline.yaml` |
| Single-node or single-device jobscript | Yes | `jobs/run_single_gcd.sh` |
| Validation command/script | Yes | `scripts/validate_run.py` |
| Expected artifact tree | Yes | `assets/expected-output-tree.txt` |
| Troubleshooting page | Yes | `troubleshooting/common-failures.md` |

## 7. Content Acceptance Criteria

- Lesson is runnable top-to-bottom without external explanation
- Main path avoids early branching
- Commands have explicit purpose
- Success artifacts are named and located clearly
- At least three failure modes are documented

## 8. Technical Acceptance Criteria

- Fresh user from onboarding can run with only path/account edits
- Uses AI Factory container path from `env.sh`
- Produces checkpoint or adapter artifact
- Produces evaluation summary
- Includes explicit GPU visibility validation

## 9. Pedagogical Acceptance Criteria

- Teaches exactly one operational capability
- Includes one controlled post-baseline modification
- Communicates distinction from training from scratch
- Leaves learner with reusable pattern confidence

## 10. Testing Plan

Record outputs for each gate:

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Data prep | `python data/prepare_sample_data.py --output data/sample_data` | `TBD` | `TBD` |
| Local dry run | `python scripts/train.py --config configs/baseline.yaml` | `TBD` | `TBD` |
| Validation | `python scripts/validate_run.py --run-dir outputs/baseline-run` | `TBD` | `TBD` |
| LUMI single GCD | `sbatch jobs/run_single_gcd.sh` | `TBD` | `TBD` |
| LUMI single node | `sbatch jobs/run_single_node.sh` | `TBD` | `TBD` |

## 11. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Platform reviewer (LUMI-G specifics) | `TBD` | `Pending` | `TBD` | `TBD` |
