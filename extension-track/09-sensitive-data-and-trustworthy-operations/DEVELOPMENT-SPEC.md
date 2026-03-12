# Development Spec Template

Use this document to lock decisions and track implementation for Lesson 09.

## 1. Lesson Identity

- Lesson id: `EXT-09`
- Title: `Data Protection, Sensitive Data, and Trustworthy AI Operations on LUMI AI Factory`
- Short nav title: `Sensitive data and trustworthy AI`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Baseline architecture to modify | A/B/C/D from Lesson 8 | `TBD` | `TBD` | `TBD` | `TBD` |
| Sensitive data class focus | personal, confidential, mixed | `TBD` | `TBD` | `TBD` | `TBD` |
| Minimization strategy | field-drop, pseudonymize, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |
| Trust gate policy | sample review, automated checks, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |
| Pilot release criteria | strict gate, staged gate | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- Sensitive-stage identification
- Data minimization and pseudonymization workflow patterns
- Logging and artifact discipline
- Trust gate definition and ownership
- Revised architecture brief

### Out Of Scope

- Full legal/compliance interpretation
- Enterprise-wide access-control architecture
- Sector-specific regulatory deep dives
- Organization-wide governance programs

## 4. Learning Outcomes

Learner can:

1. identify sensitive points in a workflow
2. redesign architecture to reduce exposure
3. apply practical minimization/pseudonymization patterns
4. define and operate a trust review gate
5. produce a governed pilot architecture brief

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When this workflow is needed
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal worked example
- F. Verification
- G. Data minimization and pseudonymization
- H. Trustworthiness gate
- I. Common failure modes
- J. Operational checklist
- K. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Baseline architecture example | Yes | `worked-example/baseline-architecture.md` |
| Sensitive-data variant | Yes | `worked-example/sensitive-data-variant.md` |
| Data-flow classification template | Yes | `templates/data-classification-table.md` |
| Trust-gate template | Yes | `templates/trust-gate-template.md` |
| Revised brief template | Yes | `templates/revised-architecture-brief.md` |
| Operational checklist artifact | Yes | `assets/trust-checklist.md` |
| Anti-patterns page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- Lesson modifies earlier architecture pattern explicitly
- Handling pattern is procedural and concrete
- At least three data/trust failure modes documented

### Pedagogical

- Learner can identify sensitive stages
- Learner can explain exposure reduction in redesigned architecture
- Learner can complete reusable trust-gated architecture brief

## 8. Testing Plan

| Gate | Command | Pass/Fail | Notes |
|---|---|---|---|
| Brief template check | `python assets/validate_revised_brief.py --brief templates/revised-architecture-brief.md` | `TBD` | `TBD` |
| Structure review | Manual walkthrough from README to templates | `TBD` | `TBD` |

## 9. Review Sign-Off

| Role | Name | Status | Date | Notes |
|---|---|---|---|---|
| Content reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Technical reviewer | `TBD` | `Pending` | `TBD` | `TBD` |
| Trust/operations reviewer | `TBD` | `Pending` | `TBD` | `TBD` |

