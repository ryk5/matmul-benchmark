#pragma once

#include "matmul_common.hpp"

namespace matmul::cpu {

// 1. Naive triple-loop implementation
inline void matmul_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t m = A.rows;
    const std::size_t n = B.cols;
    const std::size_t k = A.cols;

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t l = 0; l < k; ++l) {
                sum += A[i][l] * B[l][j];
            }
            C[i][j] = sum;
        }
    }
}

// 2. Cache-friendly blocked (tiled) implementation
inline void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C, std::size_t block_size = 64) {
    const std::size_t m = A.rows;
    const std::size_t n = B.cols;
    const std::size_t k = A.cols;

    // Zero-initialize C
    std::fill(C.data.begin(), C.data.end(), 0.0f);

    for (std::size_t ii = 0; ii < m; ii += block_size) {
        for (std::size_t jj = 0; jj < n; jj += block_size) {
            for (std::size_t ll = 0; ll < k; ll += block_size) {
                const std::size_t i_end = std::min(ii + block_size, m);
                const std::size_t j_end = std::min(jj + block_size, n);
                const std::size_t l_end = std::min(ll + block_size, k);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t j = jj; j < j_end; ++j) {
                        float sum = C[i][j];
                        for (std::size_t l = ll; l < l_end; ++l) {
                            sum += A[i][l] * B[l][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

// 3. SIMD + OpenMP parallel blocked implementation
inline void matmul_simd_openmp(const Matrix& A, const Matrix& B, Matrix& C, std::size_t block_size = 64) {
    const std::size_t m = A.rows;
    const std::size_t n = B.cols;
    const std::size_t k = A.cols;

    std::fill(C.data.begin(), C.data.end(), 0.0f);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (std::size_t ii = 0; ii < m; ii += block_size) {
        for (std::size_t jj = 0; jj < n; jj += block_size) {
            for (std::size_t ll = 0; ll < k; ll += block_size) {
                const std::size_t i_end = std::min(ii + block_size, m);
                const std::size_t j_end = std::min(jj + block_size, n);
                const std::size_t l_end = std::min(ll + block_size, k);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t l = ll; l < l_end; ++l) {
                        const float a_val = A[i][l];
#ifdef _OPENMP
    #pragma omp simd
#endif
                        for (std::size_t j = jj; j < j_end; ++j) {
                            C[i][j] += a_val * B[l][j];
                        }
                    }
                }
            }
        }
    }
}

}  // namespace matmul::cpu
