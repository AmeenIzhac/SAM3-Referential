#!/usr/bin/env python3
"""Effective PCIe bandwidth per GPU (pinned H2D/D2H), plus what that implies for
a DDP gradient all-reduce of the SAM3 model. Small buffers, runs in seconds."""
import sys, time
import torch

SIZE_MB = 512
GRAD_GB = 3.37          # SAM3 trainable params in fp32 (~843M x 4 B)


def bw(dev, n=6):
    torch.cuda.set_device(dev)
    host = torch.empty(SIZE_MB * 1024 * 1024 // 4, dtype=torch.float32).pin_memory()
    d = torch.empty_like(host, device=f"cuda:{dev}")
    for _ in range(2):                      # warm up
        d.copy_(host, non_blocking=True)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        d.copy_(host, non_blocking=True)
    torch.cuda.synchronize(dev)
    h2d = SIZE_MB / 1024 * n / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(n):
        host.copy_(d, non_blocking=True)
    torch.cuda.synchronize(dev)
    d2h = SIZE_MB / 1024 * n / (time.perf_counter() - t0)
    del d, host
    torch.cuda.empty_cache()
    return h2d, d2h


if __name__ == "__main__":
    devs = [int(x) for x in (sys.argv[1:] or ["0", "1", "2"])]
    print(f"{'gpu':>4} {'H2D GB/s':>9} {'D2H GB/s':>9}  "
          f"{'ring all-reduce of 3.37 GB grads':>34}")
    for d in devs:
        h, w = bw(d)
        slow = min(h, w)
        # ring all-reduce: every rank pushes 2(N-1)/N x S bytes through its own link
        t2 = 1.0 * GRAD_GB / slow          # N=2
        t3 = (4 / 3) * GRAD_GB / slow      # N=3
        print(f"{d:>4} {h:9.2f} {w:9.2f}   N=2: {t2:6.2f} s   N=3: {t3:6.2f} s")
