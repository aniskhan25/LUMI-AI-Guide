# Partition Cheat Sheet (Operational)

## `dev-g`

- intended for debugging and quick tests
- strict runtime and node-hour limits
- not a production partition

## `LUMI-G`

- primary GPU partition for serious AI workloads
- suitable for baseline and scaled runs

## Rule

Select partition by stage purpose:

- debug stage -> `dev-g`
- baseline and production-style stages -> `LUMI-G`
