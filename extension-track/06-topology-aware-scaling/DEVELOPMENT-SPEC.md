# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 06.

## 1. Lesson Identity

- Lesson id: `EXT-06`
- Title: `Topology-Aware Scaling of Advanced AI Workloads on LUMI-G`
- Short nav title: `Scaling on MI250X`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary workload | compact fine-tuning, embedding pipeline | `TBD` | `TBD` | `TBD` | `TBD` |
| Distributed launch path | torchrun, deepspeed, other | `torchrun` | Single clear path | `TBD` | `TBD` |
| Comparison ladder | 1→8 GCD, 1→8→16, other | `TBD` | `TBD` | `TBD` | `TBD` |
| Placement policy | default, explicit bind/distribution | `TBD` | `TBD` | `TBD` | `TBD` |
| Metric set | throughput/wall-time/efficiency | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- Intra-node scaling over 8 GPU-visible GCDs
- Small multi-node scaling
- Placement metadata capture
- Throughput/speedup/efficiency comparison

### Out Of Scope

- General MPI course material
- Low-level ROCm kernel tuning
- Broad distributed-framework benchmark surveys
- Cluster-wide scheduling strategy

## 4. Learning Outcomes

Learner can:

1. decide when to stay single-device vs scale
2. launch multi-device and multi-node runs with explicit mapping
3. collect placement-aware run records
4. compare scaling efficiency against baseline
5. diagnose common topology-related scaling issues

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal working example
- F. Verification
- G. LUMI-G topology that matters
- H. Binding and distribution choices
- I. Measuring scaling
- J. Comparing configurations
- K. Common failure modes
- L. Operational checklist
- M. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Workload runner | Yes | `scripts/run_workload.py` |
| Placement inspector | Yes | `scripts/inspect_placement.py` |
| Metrics collector | Yes | `scripts/collect_metrics.py` |
| Comparison script | Yes | `scripts/compare_scaling.py` |
| Baseline config | Yes | `configs/baseline.yaml` |
| Single-node config | Yes | `configs/single_node.yaml` |
| Multi-node config | Yes | `configs/multi_node.yaml` |
| Canonical jobs | Yes | `jobs/run_1gcd.sh`, `jobs/run_8gcd_single_node.sh`, `jobs/run_multi_node.sh` |
| Troubleshooting page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- Scaling ladder is clear and minimal
- Success conditions are explicit
- Topology relevance is operationally explained

### Technical

- Run records include device/rank/binding context
- Summaries and comparisons are reproducible
- Baseline and scaled runs are comparable

### Pedagogical

- Teaches one capability: topology-aware scaling
- Learner can explain why scaled run improved or regressed
- Learner can separate resource count from placement quality

## 8. Testing Plan

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Baseline local smoke | `python scripts/run_workload.py --config configs/baseline.yaml` | `TBD` | `TBD` |
| Placement inspection | `python scripts/inspect_placement.py --config configs/baseline.yaml` | `TBD` | `TBD` |
| Collect metrics | `python scripts/collect_metrics.py --config configs/baseline.yaml` | `TBD` | `TBD` |
| Compare scaling | `python scripts/compare_scaling.py --compare-config configs/compare.yaml` | `TBD` | `TBD` |
| LUMI 1 GCD | `sbatch jobs/run_1gcd.sh` | `TBD` | `TBD` |
| LUMI 8 GCD | `sbatch jobs/run_8gcd_single_node.sh` | `TBD` | `TBD` |
| LUMI multi-node | `sbatch jobs/run_multi_node.sh` | `TBD` | `TBD` |

## 9. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Platform reviewer (topology focus) | `TBD` | `Pending` | `TBD` | `TBD` |

