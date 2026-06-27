#!/usr/bin/env python3
"""Benchmark matrix multiplication with NumPy and PyTorch."""

import argparse
import csv
import os
import time
import warnings
from typing import Callable, List

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in.*")


def benchmark(name: str, fn: Callable, A, B, ref: np.ndarray, sizes: tuple, device: str, warmup: int = 3, runs: int = 10) -> dict:
    m, k, n = sizes
    # Warmup
    for _ in range(warmup):
        fn(A, B)

    # Correctness
    if device == "cpu":
        result = np.array(fn(A, B))
    else:
        result = np.array(fn(A, B).cpu())
    max_diff = float(np.max(np.abs(result - ref)))
    if max_diff > 1e-2:
        print(f"WARNING: {name} failed correctness check (max diff = {max_diff:.4e})")

    times = []
    for _ in range(runs):
        if device == "cuda":
            import torch
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(A, B)
        if device == "cuda":
            import torch
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    min_t = times[0]
    mean = sum(times) / len(times)
    ops = 2.0 * m * n * k
    gflops = ops / (median * 1e6)
    bytes_ = 4.0 * (m * k + k * n + m * n)
    gbs = bytes_ / (median * 1e6)

    return {
        "name": name,
        "m": m,
        "k": k,
        "n": n,
        "median_ms": median,
        "min_ms": min_t,
        "mean_ms": mean,
        "gflops": gflops,
        "gbyte_s": gbs,
        "max_diff": max_diff,
    }


def numpy_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def run_numpy(size: int, seed: int = 1) -> dict:
    np.random.seed(seed)
    scale = 1.0 / size
    A = (np.random.rand(size, size).astype(np.float32) - 0.5) * scale
    B = (np.random.rand(size, size).astype(np.float32) - 0.5) * scale
    ref = A @ B
    return benchmark("numpy", numpy_matmul, A, B, ref, (size, size, size), "cpu")


def run_pytorch_cpu(size: int, seed: int = 1) -> dict:
    import torch
    torch.manual_seed(seed)
    scale = 1.0 / size
    A = (torch.rand(size, size, dtype=torch.float32) - 0.5) * scale
    B = (torch.rand(size, size, dtype=torch.float32) - 0.5) * scale
    ref = (A @ B).numpy()
    return benchmark("pytorch_cpu", lambda a, b: a @ b, A, B, ref, (size, size, size), "cpu")


def run_pytorch_cuda(size: int, seed: int = 1) -> dict:
    import torch
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(seed)
    scale = 1.0 / size
    A = ((torch.rand(size, size, dtype=torch.float32, device="cuda") - 0.5) * scale)
    B = ((torch.rand(size, size, dtype=torch.float32, device="cuda") - 0.5) * scale)
    ref = (A @ B).cpu().numpy()
    return benchmark("pytorch_cuda", lambda a, b: a @ b, A, B, ref, (size, size, size), "cuda")


def main():
    parser = argparse.ArgumentParser(description="Benchmark NumPy/PyTorch matmul")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--output", default="results/python_results.csv")
    parser.add_argument("--device", choices=["cpu", "cuda", "all"], default="all")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    backends = []
    if args.device in ("cpu", "all"):
        backends.append(("numpy", run_numpy))
        backends.append(("pytorch_cpu", run_pytorch_cpu))
    if args.device in ("cuda", "all"):
        backends.append(("pytorch_cuda", run_pytorch_cuda))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "m", "k", "n", "median_ms", "min_ms", "mean_ms", "gflops", "gbyte_s", "max_diff"])
        writer.writeheader()

        for size in args.sizes:
            print(f"\n=== Benchmarking {size}x{size}x{size} ===")
            for name, fn in backends:
                try:
                    result = fn(size)
                    if result is None:
                        print(f"Skipping {name}: not available")
                        continue
                    writer.writerow(result)
                    print(f"{name:20s} median_ms={result['median_ms']:8.3f} gflops={result['gflops']:8.2f} gbyte/s={result['gbyte_s']:8.2f}")
                except Exception as e:
                    print(f"ERROR running {name}: {e}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
