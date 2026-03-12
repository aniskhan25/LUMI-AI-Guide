# Development Spec Template

Use this file to lock implementation choices for Lesson 11.

## 1. Lesson Identity

- Lesson id: `EXT-11`
- Title: `Team Operating Models and Collaboration Patterns for AI Factory Projects`
- Short nav title: `Team collaboration on AI Factory`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Team model | single-team, split-role, provider-consumer | `TBD` | `TBD` | `TBD` | `TBD` |
| Authoritative dataset location | project workspace, LUMI-O, DaaS + local copy | `TBD` | `TBD` | `TBD` | `TBD` |
| Promotion reviewer role | evaluator, lead, rotating approver | `TBD` | `TBD` | `TBD` | `TBD` |
| Cross-project sharing scope | full bucket, prefix-scoped, artifact-only | `TBD` | `TBD` | `TBD` | `TBD` |
| Handoff package minimum | docs only, docs + artifacts, full reproducible bundle | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- role ownership and review responsibilities
- artifact class boundaries and handoffs
- project-internal vs cross-project sharing patterns
- practical use of LUMI-O and DaaS positioning

### Out Of Scope

- enterprise IAM architecture
- legal contract structures
- broad project-management methodology
- organization-wide governance frameworks

## 4. Learning Outcomes

Learner can:

1. define role ownership for key artifact classes
2. distinguish source-of-truth vs working dataset copies
3. attach review gates to promoted outputs
4. produce a usable handoff checklist
5. choose collaboration pattern that matches project scale

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal worked example
- F. Verification
- G. Choosing team operating model
- H. Shared data and artifact patterns
- I. Review and handoff rules
- J. Common failure modes
- K. Operational checklist
- L. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Team operating model template | Yes | `templates/team-operating-model.md` |
| Responsibility matrix template | Yes | `templates/responsibility-matrix.md` |
| Artifact flow template | Yes | `templates/artifact-flow-map.md` |
| Handoff checklist template | Yes | `templates/handoff-checklist.md` |
| Worked example | Yes | `worked-example/*.md` |
| Anti-patterns page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- one realistic multi-person scenario
- explicit owner and reviewer mapping
- explicit source-of-truth dataset definition
- at least three collaboration failure modes

### Technical

- LUMI-O sharing positioning is accurate for project and cross-project collaboration
- DaaS is positioned as curated upstream source, not generic storage
- flow remains consistent with Lesson 10 lifecycle states

### Pedagogical

- learner can explain artifact ownership map
- learner can explain internal vs cross-project sharing difference
- learner can produce a reusable collaboration blueprint
