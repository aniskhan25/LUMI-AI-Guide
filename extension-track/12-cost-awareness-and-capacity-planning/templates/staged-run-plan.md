# Staged Run Plan Template

## Stage Ladder

| Stage | Purpose | Partition | Resources | Walltime | Expected Output | Decision Gate |
|---|---|---|---|---|---|---|
| Stage 0 | Debug and smoke test | dev-g | 1 GCD | <= 3h | command and path validation | proceed / fix |
| Stage 1 | Baseline measurement | LUMI-G | 1 GCD | `TBD` | baseline throughput/quality | proceed / stop |
| Stage 2 | Single-node comparison | LUMI-G | up to 1 node | `TBD` | throughput delta | proceed / hold |
| Stage 3 | Multi-node trial | LUMI-G | `TBD` nodes | `TBD` | scaled throughput and efficiency | promote / reject |

## Rules

- no scale-up without Stage 1 baseline
- compare like-for-like effective workload
- record measured output before moving to next stage
