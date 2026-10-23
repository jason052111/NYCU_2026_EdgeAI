#pragma once
// =============================================================================
//  custom.cuh  --  Your own optimized kernels (Part 4)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// e.g. a vectorized / double-buffered / swizzled variant of the reg2d kernel,
// a shape-dispatching launcher, ... Register each one in src/kernels.cuh.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_custom_pad4(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM+4];
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  const int row_base = blockIdx.y * BM;
  const int col_base = blockIdx.x * BN;
  for (int k0 = 0; k0 < K; k0 += BK) {
    // TODO(Part 3b): load the BM x BK tile of A into As (transposed) and the
    // BK x BN tile of B into Bs. There are NT threads but BM*BK and BK*BN
    // elements, so each thread must load several elements.
    for (int idx = tid; idx < BM * BK; idx += NT) {
      const int m = idx / BK;
      const int k = idx % BK;

      As[k][m] = A[(size_t)(row_base + m) * K + (k0 + k)];
    }

    for (int idx = tid; idx < BK * BN; idx += NT) {
      const int k = idx / BN;
      const int n = idx % BN;

      Bs[k][n] = B[(size_t)(k0 + k) * N + (col_base + n)];
    }

    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  // TODO(Part 3b): write acc[i][j] to C:  C = alpha * acc + beta * C
  for (int i = 0; i < TM; ++i) {
    const int row = row_base + tr * TM + i;

    for (int j = 0; j < TN; ++j) {
      const int col = col_base + tc * TN + j;
      const size_t idx = (size_t)row * N + col;

      C[idx] = alpha * acc[i][j] + beta * C[idx];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_custom_pad4(int M, int N, int K, float alpha,
                        const float* A, const float* B,
                        float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0,
                "tile must divide block tile");

  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");

  if (M % BM || N % BN || K % BK) return false;

  dim3 grid(N / BN, M / BM);
  sgemm_custom_pad4<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);

  return true;
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_custom_pad4_vec4(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM+4];
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  const int row_base = blockIdx.y * BM;
  const int col_base = blockIdx.x * BN;
  for (int k0 = 0; k0 < K; k0 += BK) {
    // TODO(Part 3b): load the BM x BK tile of A into As (transposed) and the
    // BK x BN tile of B into Bs. There are NT threads but BM*BK and BK*BN
    // elements, so each thread must load several elements.
    static_assert(BK % 4 == 0, "BK must be divisible by 4");

    for (int idx = tid; idx < BM * (BK / 4); idx += NT) {
    const int m = idx / (BK / 4);
    const int k = (idx % (BK / 4)) * 4;

    const float4 v = *reinterpret_cast<const float4*>(
        A + (size_t)(row_base + m) * K + (k0 + k));

    As[k + 0][m] = v.x;
    As[k + 1][m] = v.y;
    As[k + 2][m] = v.z;
    As[k + 3][m] = v.w;
    }

    static_assert(BN % 4 == 0, "BN must be divisible by 4");

    for (int idx = tid; idx < BK * (BN / 4); idx += NT) {
    const int k = idx / (BN / 4);
    const int n = (idx % (BN / 4)) * 4;

    const float4 v = *reinterpret_cast<const float4*>(
        B + (size_t)(k0 + k) * N + (col_base + n));

    Bs[k][n + 0] = v.x;
    Bs[k][n + 1] = v.y;
    Bs[k][n + 2] = v.z;
    Bs[k][n + 3] = v.w;
    }

    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  // TODO(Part 3b): write acc[i][j] to C:  C = alpha * acc + beta * C
  for (int i = 0; i < TM; ++i) {
    const int row = row_base + tr * TM + i;

    for (int j = 0; j < TN; ++j) {
      const int col = col_base + tc * TN + j;
      const size_t idx = (size_t)row * N + col;

      C[idx] = alpha * acc[i][j] + beta * C[idx];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_custom_pad4_vec4(int M, int N, int K, float alpha,
                             const float* A, const float* B,
                             float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0,
                "tile must divide block tile");
  static_assert(BK % 4 == 0 && BN % 4 == 0,
                "BK and BN must be divisible by 4");

  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");

  if (M % BM || N % BN || K % BK) return false;

  dim3 grid(N / BN, M / BM);
  sgemm_custom_pad4_vec4<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);

  return true;
}