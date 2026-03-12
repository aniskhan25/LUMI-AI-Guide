# Common Planning Failures

## 1) Scaling Before Baseline

Symptoms:

- multi-node job launched before single-device measurement
- no comparable baseline for decision-making

Fix:

- require baseline stage completion before scale-up stages

## 2) Using `dev-g` As Production

Symptoms:

- repeat operational runs executed in debug partition
- unstable or constrained scheduling behavior

Fix:

- keep `dev-g` for quick validation only
- move production-style runs to LUMI-G

## 3) Recomputation Waste

Symptoms:

- expensive preprocessing rerun repeatedly
- avoidable duplicate model preparation

Fix:

- preserve and version reusable artifacts
- define explicit recomputation triggers

## 4) Invalid Run Comparisons

Symptoms:

- runs compared with different effective workloads
- misleading throughput conclusions

Fix:

- normalize workload across compared stages
- document comparison assumptions

## 5) Missing Stop Criteria

Symptoms:

- scaling continues without clear value threshold
- resources consumed without decision progress

Fix:

- define stop and hold criteria before stage launch
