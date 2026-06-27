#include "matmul_common.hpp"
#include "matmul_cpu.hpp"

#ifdef BLAS_FOUND
#include "matmul_blas.hpp"
#endif

#ifdef EIGEN3_FOUND
#include "matmul_eigen.hpp"
#endif

#include <functional>
#include <numeric>
#include <vector>
#include <fstream>

namespace matmul {

using MatmulFunc = std::function<void(const Matrix&, const Matrix&, Matrix&)>;

BenchmarkResult run_benchmark(const std::string& name,
                              std::size_t m, std::size_t n, std::size_t k,
                              const MatmulFunc& func,
                              const Matrix& A, const Matrix& B,
                              const Matrix& reference,
                              int warmup_runs = 2,
                              int timed_runs = 5) {
    Matrix C(m, n);

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        func(A, B, C);
    }

    // Verify correctness
    if (!reference.data.empty()) {
        func(A, B, C);
        if (!check_result(C, reference, 1e-3)) {
            std::cerr << "WARNING: " << name << " failed correctness check\n";
        }
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        Timer t;
        func(A, B, C);
        times.push_back(t.elapsed_ms());
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
        sizes = {256, 512, 1024, 2048, 4096};
    }

    std::vector<BenchmarkResult> results;
    std::ofstream csv("results/cpu_results.csv");
    csv << "name,m,k,n,median_ms,min_ms,mean_ms,gflops,gbyte_s\n";

    for (std::size_t size : sizes) {
        const std::size_t m = size;
        const std::size_t n = size;
        const std::size_t k = size;

        std::cout << "\n=== Benchmarking " << m << "x" << k << "x" << n << " ===\n";

        Matrix A = Matrix::random(m, k, 1, 1.0f / static_cast<float>(m));
        Matrix B = Matrix::random(k, n, 2, 1.0f / static_cast<float>(k));
        Matrix C_ref(m, n);

        // Reference result from BLAS or blocked
#ifdef BLAS_FOUND
        cpu::matmul_blas(A, B, C_ref);
        const std::string ref_name = "blas";
#else
        cpu::matmul_blocked(A, B, C_ref);
        const std::string ref_name = "blocked";
#endif

        std::cout << "Reference: " << ref_name << "\n";

        // Naive
        {
            auto r = run_benchmark("cpu_naive", m, n, k, cpu::matmul_naive, A, B, C_ref, 1, 3);
            results.push_back(r);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }

        // Blocked
        {
            auto r = run_benchmark("cpu_blocked", m, n, k,
                [](const Matrix& A, const Matrix& B, Matrix& C) { cpu::matmul_blocked(A, B, C); },
                A, B, C_ref, 2, 5);
            results.push_back(r);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }

        // SIMD + OpenMP
        {
            auto r = run_benchmark("cpu_simd_openmp", m, n, k,
                [](const Matrix& A, const Matrix& B, Matrix& C) { cpu::matmul_simd_openmp(A, B, C); },
                A, B, C_ref, 2, 5);
            results.push_back(r);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }

#ifdef BLAS_FOUND
        {
            auto r = run_benchmark("cpu_blas", m, n, k, cpu::matmul_blas, A, B, C_ref, 2, 5);
            results.push_back(r);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }
#endif

#ifdef EIGEN3_FOUND
        {
            auto r = run_benchmark("cpu_eigen", m, n, k, cpu::matmul_eigen, A, B, C_ref, 2, 5);
            results.push_back(r);
            print_result(r);
            csv << r.name << "," << r.m << "," << r.k << "," << r.n << ","
                << r.median_ms << "," << r.min_ms << "," << r.mean_ms << ","
                << r.gflops << "," << r.bandwidth_gbs << "\n";
        }
#endif
    }

    csv.close();
    std::cout << "\nResults saved to results/cpu_results.csv\n";
    return 0;
}
