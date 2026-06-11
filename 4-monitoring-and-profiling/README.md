# 4. Monitoring and Profiling

## Monitoring with `rocm-smi`

While a job is running, attach to its node and check GPU utilisation:

```bash
srun --jobid <jobid> --interactive --pty /bin/bash
watch -n1 rocm-smi
```

**What to look at:** GPU utilisation and memory are shown, but the most reliable indicator of full GPU use is **power draw**. A single GCD fully loaded draws ~300W; a full MI250x (2 GCDs) draws ~500W. High utilisation with low power usually means the GPU is stalled waiting for data.

## PyTorch profiler

For code-level bottlenecks, use the PyTorch profiler to generate a trace:

```bash
sbatch run_profiled.sh
```

This produces `trace.json` in the lesson directory. Load it at [ui.perfetto.dev](https://ui.perfetto.dev/) to inspect time spent per operation.

The key profiler pattern in `visiontransformer_profiled.py`:

```python
from torch.profiler import profile, ProfilerActivity

prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
prof.start()

# ... training steps to profile ...

prof.stop()
prof.export_chrome_trace("trace.json")
```

Profile a small slice of the training loop (a single epoch or 10% of batches) — trace files grow quickly and browsers are typically limited to ~2 GB.

Perfetto navigation: `W`/`S` to zoom, `A`/`D` to move.

## Hardware-level profiling

For deeper hardware-level analysis (memory bandwidth, cache hit rates, instruction throughput), LUMI provides AMD's `rocprof`, `Omnitrace`, and `Omniperf`. See the [LUMI training materials](https://lumi-supercomputer.github.io/LUMI-training-materials/) for walkthroughs.

## Next

[5. Experiment Tracking](../5-experiment-tracking/README.md)
