import os
import time

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

if rank == 0:
    print(f"torch {torch.__version__}, rccl {torch.cuda.nccl.version()}, "
          f"FI_MR_CACHE_MONITOR={os.environ.get('FI_MR_CACHE_MONITOR', '<unset>')}, ranks {world}")
    print(f"{'size_MB':>10} {'time_ms':>10} {'busbw_GBs':>10}")

for exp in range(20, 31, 2):  # 1 MiB .. 1 GiB
    nbytes = 2**exp
    x = torch.ones(nbytes // 4, dtype=torch.float32, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t = (time.perf_counter() - start) / iters

    busbw = nbytes * 2 * (world - 1) / world / t / 1e9
    if rank == 0:
        print(f"{nbytes / 2**20:>10.0f} {t * 1e3:>10.3f} {busbw:>10.2f}")

dist.destroy_process_group()
