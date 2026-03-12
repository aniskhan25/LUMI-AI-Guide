# Common Mis-Architectures

## 1) Adapting When Retrieval Would Solve the Problem

Pattern:

- Team starts model adaptation even though failures are mainly stale/missing knowledge in documents.

Correction:

- Prefer Retrieval-Grounded Assistant first, then adapt only if task behavior still needs specialization.

## 2) Building RAG for a Pure Throughput Transformation Task

Pattern:

- Team adds retrieval/generation complexity to a workload that only needs large-scale deterministic transformation.

Correction:

- Use High-Throughput Inference Factory pattern.

## 3) Scaling Infrastructure Before Baseline Evaluation

Pattern:

- Team scales to multi-node workflows without quality baseline or failure taxonomy.

Correction:

- Run Evaluate-and-Improve Loop first.

## 4) Adding Serving Complexity Too Early

Pattern:

- Team designs endpoint/service layers before request schema and batch path are stable.

Correction:

- Stabilize batch/service-style inference workflow first, then decide endpoint lifecycle scope.

## 5) Forcing Every Component Onto LUMI-G

Pattern:

- All system components are placed on GPU nodes regardless of computational need.

Correction:

- Place compute-heavy model stages on LUMI-G.
- Keep lighter orchestration/reporting components minimal and appropriately placed.

