# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 05.

## 1. Lesson Identity

- Lesson id: `EXT-05`
- Title: `Synthetic Data and Data-Centric AI Workflows on LUMI-G`
- Short nav title: `Synthetic data on MI250X`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary downstream workflow | RAG QA, classification, extraction | `RAG QA` | Direct continuity with Lessons 3-4 | `TBD` | `TBD` |
| Weak-case source | Lesson 4 failures, manual slice, mixed | `TBD` | `TBD` | `TBD` | `TBD` |
| Synthetic generation mode | template-guided, model-guided, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |
| Filtering policy | rule-based, model-based, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |
| Comparison criterion | coverage delta, score delta, weighted mix | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- Gap identification from measured weak cases
- Candidate generation
- Filtering and curation
- Augmented dataset build with provenance
- Baseline vs augmented rerun and comparison

### Out Of Scope

- Full governance and legal compliance programs
- Enterprise-scale labeling platforms
- Fully autonomous data flywheel systems
- Large multimodal synthetic pipelines

## 4. Learning Outcomes

Learner can:

1. target synthetic generation at measured weaknesses
2. filter and curate synthetic candidates with explicit criteria
3. preserve provenance through dataset augmentation
4. compare downstream results before and after augmentation
5. decide whether synthetic data improved workflow quality

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal working example
- F. Verification
- G. What makes synthetic data useful
- H. Filtering and quality control
- I. Baseline vs augmented comparison
- J. Common failure modes
- K. Operational checklist
- L. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Weak-case dataset | Yes | `data/weak_cases.jsonl` |
| Baseline dataset | Yes | `data/baseline_dataset.jsonl` |
| Generation script | Yes | `scripts/generate_candidates.py` |
| Filtering script | Yes | `scripts/filter_candidates.py` |
| Merge script | Yes | `scripts/merge_augmented_dataset.py` |
| Downstream rerun script | Yes | `scripts/rerun_downstream_task.py` |
| Comparison/report script | Yes | `scripts/compare_results.py` |
| Canonical batch jobscript | Yes | `jobs/run_synthdata_single_node.sh` |
| Troubleshooting page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- Golden path is clear and procedural
- Filtering is mandatory in the main path
- Before/after comparison is explicit
- Success artifacts are named clearly

### Technical

- Candidate, filtered, and augmented artifacts are produced
- Accepted set is subset of candidate set
- Dataset provenance is preserved
- Baseline and augmented rerun outputs are comparable
- GPU generation visibility is confirmable

### Pedagogical

- Teaches one operational capability: targeted synthetic data loop
- Learner can explain why augmentation is targeted
- Learner can distinguish generated vs accepted data

## 8. Testing Plan

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Identify weak cases | `python scripts/identify_weak_cases.py --generate-config configs/generate.yaml` | `TBD` | `TBD` |
| Generate candidates | `python scripts/generate_candidates.py --generate-config configs/generate.yaml` | `TBD` | `TBD` |
| Filter candidates | `python scripts/filter_candidates.py --generate-config configs/generate.yaml --filter-config configs/filter.yaml` | `TBD` | `TBD` |
| Merge augmented dataset | `python scripts/merge_augmented_dataset.py --generate-config configs/generate.yaml --filter-config configs/filter.yaml` | `TBD` | `TBD` |
| Rerun downstream | `python scripts/rerun_downstream_task.py --generate-config configs/generate.yaml --compare-config configs/compare.yaml` | `TBD` | `TBD` |
| Compare/report | `python scripts/compare_results.py --generate-config configs/generate.yaml --compare-config configs/compare.yaml` | `TBD` | `TBD` |
| LUMI run | `sbatch jobs/run_synthdata_single_node.sh` | `TBD` | `TBD` |

## 9. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Platform reviewer (LUMI-G data-centric focus) | `TBD` | `Pending` | `TBD` | `TBD` |

