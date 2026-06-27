#!/bin/bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build"
RESULTS_DIR="$ROOT/results"

mkdir -p "$BUILD_DIR" "$RESULTS_DIR"

echo "=== Configuring build ==="
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "=== Building GPU benchmark ==="
cmake --build "$BUILD_DIR" --target gpu_benchmark

echo "=== Running GPU benchmark ==="
"$BUILD_DIR/src/gpu/gpu_benchmark"

echo "=== Running PyTorch CUDA benchmark ==="
python3 "$ROOT/python/benchmark.py" --device cuda --output "$RESULTS_DIR/python_cuda_results.csv"

echo "=== Generating plots ==="
python3 "$ROOT/python/plot.py" \
    --input "$RESULTS_DIR/cpu_results.csv" \
          "$RESULTS_DIR/gpu_results.csv" \
          "$RESULTS_DIR/python_results.csv" \
          "$RESULTS_DIR/python_cuda_results.csv" \
    --output-dir "$RESULTS_DIR"

echo "Done. Results in $RESULTS_DIR"
