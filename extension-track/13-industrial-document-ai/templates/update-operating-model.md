# Update Operating Model Template

## 1) Ownership

- corpus update owner:
- evaluation owner:
- promotion approver:
- delivery owner:

## 2) Update Triggers

- new document revision arrives
- document approval state changes
- failure cluster appears in evaluation
- scheduled refresh window

## 3) Update Pipeline

1. ingest changed documents
2. update chunk metadata and revision links
3. re-embed affected chunks
4. rerun targeted evaluation slice
5. approve and promote updated workflow

## 4) Promotion Rules

- required artifacts:
- required evaluation evidence:
- rollback target:

## 5) Communication and Handoff

- consumer teams:
- handoff package contents:
- release note requirements:
