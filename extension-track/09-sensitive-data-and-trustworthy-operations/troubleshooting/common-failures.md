# Common Failures

## 1) Keeping baseline data flow unchanged

Symptoms:

- Sensitive-data variant has same artifacts/logs as baseline.

Fix:

- explicitly redesign each high-risk stage using the classification table.

## 2) Logging raw sensitive content by default

Symptoms:

- full prompts/responses retained in routine logs.

Fix:

- switch to ID/status/timing logging and restrict raw-text review windows.

## 3) Pseudonymization without separation of lookup data

Symptoms:

- pseudonymous IDs are mixed with reversible mapping in same artifact set.

Fix:

- separate lookup mapping from operational inference artifacts.

## 4) Trust gate defined but not operationalized

Symptoms:

- checklist exists but no owner, sample size, or pass criteria.

Fix:

- complete trust-gate template with explicit ownership and thresholds.

## 5) Treating private data source as sufficient control

Symptoms:

- assumption that restricted storage alone guarantees trustworthy operation.

Fix:

- apply workflow-level minimization, logging discipline, and output review gate.

