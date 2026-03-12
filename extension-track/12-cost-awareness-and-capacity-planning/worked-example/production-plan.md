# Worked Example: Production Plan

## Chosen Pattern

Single-node steady-state batch pattern on LUMI-G with periodic scale bursts.

## Why This Pattern

- baseline and one-node stages met throughput target
- multi-node gain did not justify regular use
- stable quality with reusable artifacts

## Operational Plan

- run daily batch on single node
- reserve multi-node only for backlog-clearing windows
- keep preprocessing artifacts cached and versioned
- review planned vs actual usage weekly

## Review Trigger

Revisit plan when:

- request volume doubles
- model configuration changes materially
- quality gate requires more expensive inference settings
