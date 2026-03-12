# Worked Example: Scaling Decision

## Comparison Path

1. Stage 1 baseline on 1 GCD
2. Stage 2 one-node run for throughput comparison
3. optional Stage 3 multi-node trial only if Stage 2 gain is meaningful

## Example Gate

Scale from Stage 2 to Stage 3 only if:

- throughput gain from Stage 1 to Stage 2 is >= 1.8x
- quality metrics do not regress
- artifact reuse is intact (no repeated heavy preprocessing)

## Hold Condition

Do not scale when:

- throughput gain is small relative to added resources
- bottleneck is I/O or preprocessing instead of model compute
