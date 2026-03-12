# Decision Matrix

Use this matrix to choose one architecture pattern quickly.

| Decision Dimension | Adapt-and-Apply | Retrieval-Grounded Assistant | Evaluate-and-Improve Loop | High-Throughput Inference Factory |
|---|---|---|---|---|
| Need model adaptation | High | Low/Optional | Optional | Low |
| Need document grounding | Low | High | Depends | Low/Optional |
| Trustworthiness iteration priority | Medium | High | Highest | Medium |
| Corpus/request scale | Medium | Medium-High | Medium | Highest |
| Throughput-first objective | Medium | Medium | Medium | Highest |
| Fastest pilot path when quality unknown | Medium | Medium | Highest | Medium |

## Selection Questions

1. Is missing domain knowledge the main issue, or missing retrieval quality?
2. Is baseline quality uncertain enough to require evaluation-first iteration?
3. Is workload mostly corpus-scale transformation where throughput dominates?
4. Do we need evidence-grounded outputs for stakeholder trust?

## Tie-Break Rule

If two patterns appear viable, prefer the one with:

- fewer components in pilot scope
- clearer evaluation gate
- faster path to customer-visible value

