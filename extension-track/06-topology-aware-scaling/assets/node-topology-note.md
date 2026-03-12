# Node Topology Note

This lesson uses a practical topology model for LUMI-G runs:

- A full node is treated as 8 GPU-visible GCD devices.
- CPU resources are organized into 4 NUMA domains.
- Rank-to-device mapping should be explicit and validated.
- Placement metadata is captured for each run in `raw/placement_rank*.json`.

Use this note as a reminder that scaling results should be interpreted with placement and communication context, not GPU count alone.

