#pragma once

#include "matmul_common.hpp"

#ifdef BLAS_FOUND
#include <cblas.h>

namespace matmul::cpu {

inline void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C) {
    const int m = static_cast<int>(A.rows);
    const int n = static_cast<int>(B.cols);
    const int k = static_cast<int>(A.cols);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A.data.data(), k,
                B.data.data(), n, beta, C.data.data(), n);
    #pragma clang diagnostic pop
}

}  // namespace matmul::cpu

#endif  // BLAS_FOUND
