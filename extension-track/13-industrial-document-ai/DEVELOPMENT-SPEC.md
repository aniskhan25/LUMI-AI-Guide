# Development Spec Template

Use this file to lock design choices and implementation scope for Lesson 13.

## 1. Lesson Identity

- Lesson id: `EXT-13`
- Title: `Industrial Document AI and Technical Knowledge Workflows on LUMI AI Factory`
- Short nav title: `Industrial document AI`
- Owner: `<team or person>`
- Target branch: `feature/advanced-onboarding-lessons`

## 2. Decision Lock (Must Complete First)

| Decision | Options | Chosen | Rationale | Owner | Date |
|---|---|---|---|---|---|
| Primary scenario | maintenance assistant, procedure assistant, report QA | `TBD` | `TBD` | `TBD` | `TBD` |
| Answer mode | concise evidence answer, guided procedural answer | `TBD` | `TBD` | `TBD` | `TBD` |
| Revision policy | latest-only, pinned-by-version, dual-track | `TBD` | `TBD` | `TBD` | `TBD` |
| Review gate strictness | strict fail-closed, risk-tiered, advisory | `TBD` | `TBD` | `TBD` | `TBD` |
| Update cadence | event-driven, periodic, hybrid | `TBD` | `TBD` | `TBD` | `TBD` |

## 3. Scope Boundaries

### In Scope

- technical document corpus design
- grounded retrieval and evidence-linked answers
- technical evaluation and review gate pattern
- update and ownership operating model

### Out Of Scope

- CAD/simulation integration
- multimodal industrial vision pipelines
- sector-specific compliance frameworks in depth
- full enterprise search platform architecture

## 4. Learning Outcomes

Learner can:

1. map industrial document problem to grounded workflow
2. define revision-aware corpus schema
3. define answer schema with evidence linkage
4. evaluate technical support quality beyond fluency
5. document update and ownership model for team use

## 5. Lesson Structure Contract

The lesson must include:

- A. What this lesson enables
- B. When to use this workflow
- C. Prerequisites
- D. Workflow at a glance
- E. Minimal worked example
- F. Verification
- G. Corpus design for technical documents
- H. Grounded answering and trust rules
- I. Evaluation for technical correctness
- J. Update and lifecycle model
- K. Common failure modes
- L. Operational checklist
- M. Next lesson

## 6. Mandatory Deliverables

| Deliverable | Required | Path |
|---|---|---|
| Lesson README | Yes | `README.md` |
| Domain use-case brief template | Yes | `templates/domain-use-case-brief.md` |
| Technical corpus schema template | Yes | `templates/technical-corpus-schema.md` |
| Evaluation checklist template | Yes | `templates/evaluation-checklist.md` |
| Update operating model template | Yes | `templates/update-operating-model.md` |
| Worked example | Yes | `worked-example/*.md` |
| Anti-pattern page | Yes | `troubleshooting/common-failures.md` |

## 7. Acceptance Criteria

### Content

- one realistic industrial document scenario
- grounding and evidence explicit throughout
- revision handling explicit
- at least three domain-specific failure modes

### Technical

- AI Software Environment, LUMI-G, LUMI-O roles are accurately positioned
- DaaS presented as curated input option
- update model links to lifecycle and team operating lessons

### Pedagogical

- learner can explain grounded knowledge workflow rationale
- learner can define authoritative source and revision model
- learner can produce reusable domain solution blueprint
