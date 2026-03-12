# Common Failures

## 1) Output IDs do not align with evaluation set

Symptoms:

- missing outputs for some `query_id`
- scoring join failures

Fix:

- preserve stable `query_id` end-to-end
- validate count and ID set before scoring

## 2) Metrics do not match customer objective

Symptoms:

- good aggregate score but obviously bad user outcomes

Fix:

- adjust required terms/rubric fields
- include domain-specific error categories
- review failure samples, not just aggregates

## 3) Variant comparison is not controlled

Symptoms:

- cannot explain why one variant changed

Fix:

- change one variable only (for example `top_k`)
- keep same eval set and scoring script
- document condition in run metadata

## 4) Missing failure review

Symptoms:

- report contains averages only

Fix:

- always run `extract_failures.py`
- include representative failed records in report

## 5) Drift between scored records and summary report

Symptoms:

- summary metrics disagree with scored records

Fix:

- regenerate report from fresh summary files
- verify `query_id` counts and variant labels

