# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 04.

## 1. Lesson Identity

- Lesson id: `EXT-04`
- Title: `Evaluation, Benchmarking, and Trustworthiness for Customer AI Workflows on LUMI-G`
- Short nav title: `Evaluation and trustworthiness on MI250X`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary workflow under evaluation | Lesson 3 RAG, Lesson 2 inference, Lesson 1 adaptation | `Lesson 3 RAG` | Direct continuity and trustworthiness focus | `TBD` | `TBD` |
| Core metric set | retrieval hit, answer score, groundedness, completion | `TBD` | `TBD` | `TBD` | `TBD` |
| Variant comparison axis | top-k, chunk size, prompt, embedding model | `TBD` | `TBD` | `TBD` | `TBD` |
| Failure taxonomy | retrieval, grounding, correctness, format | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- Versioned benchmark set
- Repeatable scoring pipeline
- Failure-case extraction
- Controlled variant comparison
- Compact decision-oriented summary report

### Out Of Scope

- Full governance and compliance frameworks
- Human-subject process design
- Large-scale red-teaming
- Enterprise production monitoring stacks

## 4. Learning Outcomes

Learner can:

1. run repeatable workflow evaluation on LUMI-G
2. distinguish system success from task success
3. inspect representative failure modes
4. compare two variants responsibly
5. preserve evaluation artifacts for change tracking

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal working example
- F. Verification
- G. What to measure and why
- H. Failure analysis
- I. Variant comparison
- J. Common failure modes
- K. Operational checklist
- L. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Evaluation dataset and schema | Yes | `data/eval_set.jsonl`, `data/expected-schema.md` |
| Baseline run wrapper | Yes | `scripts/run_baseline_eval.py` |
| Scoring script | Yes | `scripts/score_outputs.py` |
| Comparison script | Yes | `scripts/compare_variants.py` |
| Failure extraction script | Yes | `scripts/extract_failures.py` |
| Report generation script | Yes | `scripts/build_report.py` |
| Canonical batch jobscript | Yes | `jobs/run_eval_single_node.sh` |
| Troubleshooting page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- Golden path is clear and minimally branched
- Success conditions are explicit
- Failure review is part of main path
- Variant comparison is included

### Technical

- Evaluation produces scored records and summary report
- IDs stay stable across artifacts
- Compute-heavy steps can run on GPU path
- Workflow is robust for repeated lesson delivery

### Pedagogical

- Teaches one operational capability: decision-oriented evaluation
- Learner can explain measured dimensions
- Learner can justify variant choice with evidence

## 8. Testing Plan

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Run baseline | `python scripts/run_baseline_eval.py --config configs/eval.yaml --variant baseline` | `TBD` | `TBD` |
| Run candidate | `python scripts/run_baseline_eval.py --config configs/eval.yaml --variant candidate` | `TBD` | `TBD` |
| Score outputs | `python scripts/score_outputs.py --config configs/eval.yaml --variant baseline` | `TBD` | `TBD` |
| Extract failures | `python scripts/extract_failures.py --config configs/eval.yaml --variant baseline` | `TBD` | `TBD` |
| Compare | `python scripts/compare_variants.py --config configs/eval.yaml` | `TBD` | `TBD` |
| Build report | `python scripts/build_report.py --config configs/eval.yaml` | `TBD` | `TBD` |
| LUMI run | `sbatch jobs/run_eval_single_node.sh` | `TBD` | `TBD` |

## 9. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Platform reviewer (LUMI-G eval focus) | `TBD` | `Pending` | `TBD` | `TBD` |

