#include "matmul_gpu.cuh"

#include <matmul_common.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace matmul {

using GpuFunc = void (*)(const float*, const float*, float*, int, int, int, cudaStream_t);

BenchmarkResult run_gpu_benchmark(const std::string& name,
                                  std::size_t m, std::size_t n, std::size_t k,
                                  GpuFunc func,
                                  const float* d_A, const float* d_B, float* d_C,
                                  const Matrix& reference,
                                  int warmup_runs = 2,
                                  int timed_runs = 10) {
    cudaStream_t stream;
    gpu::CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        func(d_A, d_B, d_C, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
    }
    gpu::CHECK_CUDA(cudaStreamSynchronize(stream));

    // Verify correctness
    Matrix C_host(m, n);
    func(d_A, d_B, d_C, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
    gpu::CHECK_CUDA(cudaStreamSynchronize(stream));
    gpu::CHECK_CUDA(cudaMemcpy(C_host.data.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    if (!reference.data.empty()) {
        if (!check_result(C_host, reference, 1e-2)) {
            std::cerr << "WARNING: " << name << " failed correctness check (max diff = "
                      << max_abs_diff(C_host, reference) << ")\n";
        }
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        gpu::CHECK_CUDA(cudaStreamSynchronize(stream));
        auto start = std::chrono::high_resolution_clock::now();
        func(d_A, d_B, d_C, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
        gpu::CHECK_CUDA(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    gpu::CHECK_CUDA(cudaStreamDestroy(stream));

    std::sort(times.begin(), times.end());
    double median = times[timed_runs / 2];
    double min_t = times.front();
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    return BenchmarkResult{
        name, m, n, k, median, min_t, mean,
        compute_gflops(m, n, k, median),
        compute_bandwidth_gbs(m, n, k, median)
    };
}

BenchmarkResult run_cublas_benchmark(const std::string& name,
                                     std::size_t m, std::size_t n, std::size_t k,
                                     cublasHandle_t handle,
                                     const float* d_A, const float* d_B, float* d_C,
                                     const Matrix& reference,
                                     int warmup_runs = 3,
                                     int timed_runs = 10) {
    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        gpu::matmul_gpu_cublas(handle, d_A, d_B, d_C,
                               static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    }
    gpu::CHECK_CUDA(cudaDeviceSynchronize());

    // Verify correctness
    Matrix C_host(m, n);
    gpu::matmul_gpu_cublas(handle, d_A, d_B, d_C,
                           static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    gpu::CHECK_CUDA(cudaDeviceSynchronize());
    gpu::CHECK_CUDA(cudaMemcpy(C_host.data.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    if (!reference.data.empty()) {
        if (!check_result(C_host, reference, 1e-2)) {
            std::cerr << "WARNING: " << name << " failed correctness check (max diff = "
                      << max_abs_diff(C_host, reference) << ")\n";
        }
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        gpu::CHECK_CUDA(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        gpu::matmul_gpu_cublas(handle, d_A, d_B, d_C,
                               static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
        gpu::CHECK_CUDA(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    double median = times[timed_runs / 2];
    double min_t = times.front();
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    return BenchmarkResult{
        name, m, n, k, median, min_t, mean,
        compute_gflops(m, n, k, median),
        compute_bandwidth_gbs(m, n, k, median)
    };
}

}  // namespace matmul

int main(int argc, char** argv) {
    using namespace matmul;

    std::vector<std::size_t> sizes;
    if (argc > 1) {
        sizes.push_back(std::stoull(argv[1]));
    } else {
        sizes = {256, 512, 1024, 2048, 4096, 8192};
    }

    std::ofstream csv("results/gpu_results.csv");
    csv << "name,m,k,n,median_ms,min_ms,mean_ms,gflops,gbyte_s\n";

    for (std::size_t size : sizes) {
        const std::size_t m = size;
        const std::size_t n = size;
        const std::size_t k = size;

        std::cout << "\n=== Benchmarking GPU " << m << "x" << k << "x" << n << " ===\n";

        Matrix A = Matrix::random(m, k, 1, 1.0f / static_cast<float>(m));
        Matrix B = Matrix::random(k, n, 2, 1.0f / static_cast<float>(k));
        Matrix C_ref(m, n);

        // Allocate GPU memory
        float *d_A, *d_B, *d_C;
        gpu::CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
        gpu::CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
        gpu::CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));
        gpu::CHECK_CUDA(cudaMemcpy(d_A, A.data.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
        gpu::CHECK_CUDA(cudaMemcpy(d_B, B.data.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));

        // Reference: cuBLAS
        cublasHandle_t handle;
        gpu::CHECK_CUBLAS(cublasCreate(&handle));
        gpu::matmul_gpu_cublas(handle, d_A, d_B, d_C,
                               static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
        gpu::CHECK_CUDA(cudaDeviceSynchronize());
        gpu::CHECK_CUDA(cudaMemcpy(C_ref.data.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

        // Naive CUDA
        {
            auto r = run_gpu_benchmark("gpu_naive", m, n, k, gpu::matmul_gpu_naive,
                                       d_A, d_B, d_C, C_ref, 1, 3);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }

        // Shared memory
        {
            auto r = run_gpu_benchmark("gpu_shared", m, n, k, gpu::matmul_gpu_shared,
                                       d_A, d_B, d_C, C_ref, 2, 5);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }

        // Optimized
        {
            auto r = run_gpu_benchmark("gpu_optimized", m, n, k, gpu::matmul_gpu_optimized,
                                       d_A, d_B, d_C, C_ref, 2, 5);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }

        // cuBLAS
        {
            auto r = run_cublas_benchmark("gpu_cublas", m, n, k, handle, d_A, d_B, d_C, C_ref, 3, 10);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }

        gpu::CHECK_CUBLAS(cublasDestroy(handle));
        gpu::CHECK_CUDA(cudaFree(d_A));
        gpu::CHECK_CUDA(cudaFree(d_B));
        gpu::CHECK_CUDA(cudaFree(d_C));
    }

    csv.close();
    std::cout << "\nResults saved to results/gpu_results.csv\n";
    return 0;
}
