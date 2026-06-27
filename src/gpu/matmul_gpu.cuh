#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

namespace matmul::gpu {

constexpr int TILE_SIZE = 32;

// 1. Naive CUDA kernel
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. Shared memory tiled kernel
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 3. Optimized CUDA kernel: 2D tiling with register blocking
// Block: 128x128 in C; each thread computes 8x8 tile; 256 threads per block.
__global__ void matmul_optimized_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int THREADS = (BM * BN) / (TM * TN);  // 256

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float thread_c[TM][TN] = {0.0f};
    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    int thread_id = threadIdx.x;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int thread_row = thread_id / (BN / TN);
    int thread_col = thread_id % (BN / TN);

    // Coalesced gmem-to-smem loading
    int load_a_row = thread_id / BK;
    int load_a_col = thread_id % BK;
    int load_b_row = thread_id / BN;
    int load_b_col = thread_id % BN;
    constexpr int stride_a = THREADS / BK;  // 32
    constexpr int stride_b = THREADS / BN;    // 2

    for (int bk = 0; bk < K; bk += BK) {
        // Load A tile: BM x BK (128 rows x 8 cols)
        for (int i = 0; i < BM; i += stride_a) {
            int a_row = block_row + load_a_row + i;
            int a_col = bk + load_a_col;
            s_a[load_a_row + i][load_a_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        // Load B tile: BK x BN (8 rows x 128 cols)
        for (int i = 0; i < BK; i += stride_b) {
            int b_row = bk + load_b_row + i;
            int b_col = block_col + load_b_col;
            s_b[load_b_row + i][load_b_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = s_a[thread_row * TM + i][k];
            }
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = s_b[k][thread_col * TN + j];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    thread_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int c_row = block_row + thread_row * TM + i;
            int c_col = block_col + thread_col * TN + j;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = thread_c[i][j];
            }
        }
    }
}

inline void matmul_gpu_naive(const float* A, const float* B, float* C,
                             int M, int N, int K, cudaStream_t stream = 0) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

inline void matmul_gpu_shared(const float* A, const float* B, float* C,
                              int M, int N, int K, cudaStream_t stream = 0) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_shared_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

inline void matmul_gpu_optimized(const float* A, const float* B, float* C,
                                 int M, int N, int K, cudaStream_t stream = 0) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int THREADS = (BM * BN) / (TM * TN);

    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_optimized_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

inline void matmul_gpu_cublas(cublasHandle_t handle,
                              const float* A, const float* B, float* C,
                              int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cuBLAS is column-major. Our row-major C=A*B is equivalent to
    // column-major C^T = B^T * A^T, which is sgemm("N","N",N,M,K,B,A,C).
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, B, N, A, K,
                             &beta, C, N));
}

}  // namespace matmul::gpu
