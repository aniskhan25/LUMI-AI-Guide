# Common Failures

## 1) More GPUs, lower throughput

Symptoms:

- scaled run slower than baseline

Fix:

- verify effective workload per run is comparable
- confirm communication overhead is not dominating
- check placement metadata before blaming framework

## 2) Bad rank/device mapping assumptions

Symptoms:

- unexpected performance variance across runs

Fix:

- inspect `placement_rank*.json`
- do not assume GPU index equals NUMA locality
- use explicit binding/distribution settings

## 3) Inconsistent effective batch between runs

Symptoms:

- invalid speedup/efficiency comparisons

Fix:

- record `samples_per_step` and `world_size`
- compare runs with controlled workload assumptions

## 4) Multi-node run launched incorrectly

Symptoms:

- world size mismatch
- distributed initialization errors

Fix:

- verify `MASTER_ADDR`, rendezvous endpoint, and node count
- ensure expected ranks are launched per node

## 5) Missing placement metadata

Symptoms:

- throughput collected but no diagnosis possible

Fix:

- always run `inspect_placement.py` alongside workload
- keep raw placement records with run summaries

