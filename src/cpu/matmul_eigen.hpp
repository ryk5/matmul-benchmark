#pragma once

#include "matmul_common.hpp"

#ifdef EIGEN3_FOUND
#include <Eigen/Core>

namespace matmul::cpu {

inline void matmul_eigen(const Matrix& A, const Matrix& B, Matrix& C) {
    using Map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using CMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    Map a(A.data.data(), A.rows, A.cols);
    Map b(B.data.data(), B.rows, B.cols);
    CMap c(C.data.data(), C.rows, C.cols);

    c.noalias() = a * b;
}

}  // namespace matmul::cpu

#endif  // EIGEN3_FOUND
