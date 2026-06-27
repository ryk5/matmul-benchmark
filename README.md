# Matrix Multiplication Benchmark

A benchmark suite comparing matrix multiplication implementations across CPU, GPU, and high-level libraries.

## Implemented Methods

### CPU (C++)
- `cpu_naive`: triple-loop reference
- `cpu_blocked`: cache-tiled C++
- `cpu_simd_openmp`: SIMD-vectorized + OpenMP parallel
- `cpu_blas`: vendor/optimized BLAS (Apple Accelerate, OpenBLAS, MKL)
- `cpu_eigen`: Eigen matrix library

### GPU (CUDA)
- `gpu_naive`: naive CUDA kernel
- `gpu_shared`: CUDA with shared-memory tiling
- `gpu_optimized`: 2D tiling + register blocking + coalesced loads
- `gpu_cublas`: NVIDIA cuBLAS

### Python
- `numpy`: NumPy `@`
- `pytorch_cpu`: PyTorch CPU
- `pytorch_cuda`: PyTorch CUDA

## Metrics

For each run we record:
- Median / min / mean latency (ms)
- GFLOPS (compute throughput)
- Effective GB/s (memory bandwidth)
- Correctness against a reference implementation

Optional: on Linux you can wrap the C++ binary with `perf stat -e cache-misses,cycles,instructions` to collect cache-level metrics.

## Build & Run

### CPU (macOS/Linux)

```bash
# Install Python deps
pip install -r python/requirements.txt

# Run everything
bash scripts/run_cpu.sh
```

To run a single size:

```bash
./build/src/cpu/cpu_benchmark 2048
```

### GPU (CUDA-capable machine)

```bash
bash scripts/run_gpu.sh
```

To run a single size:

```bash
./build/src/gpu/gpu_benchmark 4096
```

## Results

All CSV and PNG outputs are written to `results/`.

## Connecting to a Remote GPU Instance

If you want to run the CUDA benchmarks on a remote GPU instance:

1. Ensure the remote has CUDA, cuBLAS, CMake, and a C++ compiler.
2. Copy this repository to the remote (`scp -r` or `git clone`).
3. Run `bash scripts/run_gpu.sh` on the remote.
4. Copy `results/gpu_results.csv` back to the local machine for combined plotting.

## Notes

- On Apple Silicon, BLAS is provided by `Accelerate.framework` (collected automatically by CMake).
- The default square sizes are 256, 512, 1024, 2048, 4096. Add larger sizes (e.g., 8192) as needed.
- Naive CPU `O(N^3)` is very slow at 4096; you may want to skip it for large sizes.
