#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace matmul {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct Matrix {
    std::size_t rows;
    std::size_t cols;
    std::vector<float> data;

    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c) {}

    float* operator[](std::size_t row) { return data.data() + row * cols; }
    const float* operator[](std::size_t row) const { return data.data() + row * cols; }

    static Matrix random(std::size_t rows, std::size_t cols, unsigned seed = 42, float scale = 1.0f) {
        Matrix m(rows, cols);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : m.data) v = dist(gen) * scale;
        return m;
    }
};

struct BenchmarkResult {
    std::string name;
    std::size_t m;
    std::size_t n;
    std::size_t k;
    double median_ms;
    double min_ms;
    double mean_ms;
    double gflops;
    double bandwidth_gbs;
};

inline double compute_gflops(std::size_t m, std::size_t n, std::size_t k, double ms) {
    // 2 * m * n * k floating point operations
    double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    return ops / (ms * 1e6);  // ms to seconds, ops to GFLOPS
}

inline double compute_bandwidth_gbs(std::size_t m, std::size_t n, std::size_t k, double ms) {
    // Read A (m*k), B (k*n), write C (m*n) once, assuming float (4 bytes)
    double bytes = 4.0 * (static_cast<double>(m) * k + static_cast<double>(k) * n + static_cast<double>(m) * n);
    return bytes / (ms * 1e6);  // GB/s
}

class Timer {
    Clock::time_point start_;

public:
    Timer() : start_(Clock::now()) {}

    double elapsed_ms() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    void reset() { start_ = Clock::now(); }
};

inline double max_abs_diff(const Matrix& a, const Matrix& b) {
    assert(a.rows == b.rows && a.cols == b.cols);
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(a.data[i] - b.data[i])));
    }
    return max_diff;
}

inline bool check_result(const Matrix& result, const Matrix& reference, double tolerance = 1e-3) {
    return max_abs_diff(result, reference) <= tolerance;
}

inline void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(30) << r.name
              << " size=" << std::setw(12) << (std::to_string(r.m) + "x" + std::to_string(r.k) + "x" + std::to_string(r.n))
              << " median_ms=" << std::setw(10) << r.median_ms
              << " min_ms=" << std::setw(10) << r.min_ms
              << " gflops=" << std::setw(10) << r.gflops
              << " gbyte/s=" << std::setw(10) << r.bandwidth_gbs
              << "\n";
}

}  // namespace matmul
