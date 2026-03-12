# Development Spec Template

Use this file to lock decisions and implementation details for Lesson 12.

## 1. Lesson Identity

- Lesson id: `EXT-12`
- Title: `Cost Awareness, Capacity Planning, and Workload Selection on LUMI-G`
- Short nav title: `Cost and capacity planning on MI250X`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary workflow to plan | RAG, inference factory, adaptation | `TBD` | `TBD` | `TBD` | `TBD` |
| Baseline stage resource shape | 1 GCD, 2 GCD, 1 node | `TBD` | `TBD` | `TBD` | `TBD` |
| Scale-up gate metric | throughput, latency, quality, blended | `TBD` | `TBD` | `TBD` | `TBD` |
| Artifact reuse policy | minimal, selective, aggressive caching | `TBD` | `TBD` | `TBD` | `TBD` |
| Stop criteria policy | hard threshold, review board, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- staged run planning
- partition and scale choice heuristics
- estimate and review artifacts
- artifact reuse and recomputation control

### Out Of Scope

- grant writing or procurement strategy
- full accounting systems
- enterprise FinOps tooling
- generic queueing theory

## 4. Learning Outcomes

Learner can:

1. define a workload profile before scaling
2. build debug/test/scaled run ladder
3. set explicit scale-up and stop criteria
4. connect measured outputs to resource decisions
5. run post-execution capacity review

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal worked example
- F. Verification
- G. Billing and resource units
- H. Choosing the right scale
- I. Partition and runtime planning
- J. Storage and recomputation choices
- K. Common failure modes
- L. Operational checklist
- M. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Workload profile template | Yes | `templates/workload-profile.md` |
| Staged run plan template | Yes | `templates/staged-run-plan.md` |
| Estimate table template | Yes | `templates/estimate-table.md` |
| Post-run review template | Yes | `templates/post-run-review.md` |
| Worked example | Yes | `worked-example/*.md` |
| Anti-patterns page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- one realistic workload example
- explicit staged run ladder
- explicit partition rationale
- at least three planning failure modes

### Technical

- billing and partition notes are accurate at operational level
- baseline stage is measurable
- debug and production-style stages are distinguished
- artifact reuse is explicit

### Pedagogical

- learner can explain why and when to scale
- learner can justify partition choice
- learner can produce reusable cost-and-capacity brief
