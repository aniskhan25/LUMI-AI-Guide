# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 10.

## 1. Lesson Identity

- Lesson id: `EXT-10`
- Title: `MLOps and Lifecycle Management for AI Factory Workflows`
- Short nav title: `MLOps on LUMI AI Factory`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary workflow to manage | RAG, inference factory, adapt-and-apply | `TBD` | `TBD` | `TBD` | `TBD` |
| Version ID format | date-based, semantic, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |
| Promotion gate style | strict, staged, risk-tiered | `TBD` | `TBD` | `TBD` | `TBD` |
| Team handoff path | project workspace, LUMI-O, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |
| Retirement policy | archive-only, archive+deprecate tags | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- run manifest discipline
- artifact and naming layout
- experiment vs promoted state separation
- promotion criteria and lifecycle states
- practical sharing/staging approach

### Out Of Scope

- full enterprise CI/CD platform design
- vendor tool surveys
- complete model registry platform comparisons
- organization-wide MLOps governance programs

## 4. Learning Outcomes

Learner can:

1. version core workflow assets coherently
2. produce complete run records
3. separate draft experiments from promoted artifacts
4. define promotion and retirement decisions
5. hand off workflows with minimal ambiguity

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal worked example
- F. Verification
- G. What to version and why
- H. Artifact layout and storage pattern
- I. Promotion and retirement rules
- J. Common failure modes
- K. Operational checklist
- L. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Run manifest template | Yes | `templates/run-manifest.yaml` |
| Artifact tree template | Yes | `templates/artifact-layout.md` |
| Lifecycle states page | Yes | `templates/lifecycle-states.md` |
| Promotion checklist | Yes | `templates/promotion-checklist.md` |
| Worked example (baseline/promoted/shareable) | Yes | `worked-example/*.md` |
| Operational checklist asset | Yes | `assets/lifecycle-checklist.md` |
| Anti-pattern page | Yes | `troubleshooting/common-failures.md` |
| Manifest validator script | Yes | `scripts/validate_manifest.py` |

## 7. Acceptance Criteria

### Content

- one familiar workflow used in worked example
- promotion rules explicit
- lifecycle states explicit
- at least three lifecycle failure modes documented

### Technical

- manifest completeness check available
- promoted artifacts trace to source run IDs
- experiment and promoted locations clearly separated

### Pedagogical

- learner can explain versioning minimum set
- learner can distinguish experiment vs promoted states
- learner can produce reusable lifecycle brief/checklist

