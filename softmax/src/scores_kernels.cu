#include "../include/scores_kernels.cuh"
#include "../include/cuda_utils.cuh"

#include <cmath>

#define TILE 16

__global__ void compute_scores_kernel(const float *Q,const float *K, float *scores, int M, int N, int d)
{
    int tid_col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_row < M && tid_col < N)
    {
        float sum = 0.0f;
        for (int feature = 0; feature < d; feature++)
        {
            float q_value = Q[tid_row * d + feature];
            float k_value = K[tid_col * d + feature];

            sum += q_value * k_value;
        }
        scores[tid_row * N + tid_col] = sum / sqrtf((float)d);
    }
}

#define TILE 16
// AI generated check needed
__global__ void compute_scores_tiled_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    float *__restrict__ scores,
    int M,
    int N,
    int d)
{
    __shared__ float Qs[TILE][64];
    __shared__ float Ks[TILE][64];

    int tx = threadIdx.x; // column inside tile
    int ty = threadIdx.y; // row inside tile

    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    // Load Q tile: 16 rows x d
    for (int feature = tx; feature < d; feature += TILE)
    {
        if (row < M)
        {
            Qs[ty][feature] = Q[row * d + feature];
        }
        else
        {
            Qs[ty][feature] = 0.0f;
        }
    }

    // Load K tile: 16 cols x d
    for (int feature = ty; feature < d; feature += TILE)
    {
        if (col < N)
        {
            Ks[tx][feature] = K[col * d + feature];
        }
        else
        {
            Ks[tx][feature] = 0.0f;
        }
    }

    __syncthreads();

    float sum = 0.0f;

    if (row < M && col < N)
    {
#pragma unroll
        for (int feature = 0; feature < 64; feature++)
        {
            sum += Qs[ty][feature] * Ks[tx][feature];
        }

        scores[row * N + col] = sum * rsqrtf((float)d);
    }
}

void launch_scores_naive(
    const float* d_query,
    const float* d_key,
    float* d_scores,
    int M,
    int N,
    int d
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    compute_scores_kernel<<<grid, block>>>(d_query, d_key, d_scores, M, N, d);
    std::cout << "--GPU Scores(naive) Execution time: " << ms << "ms --\n";
    CUDA_CHECK(cudaGetLastError());
}

void launch_scores_tiled(
    const float* d_query,
    const float* d_key,
    float* d_scores,
    int M,
    int N,
    int d
) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    compute_scores_tiled_kernel<<<grid, block>>>(d_query, d_key, d_scores, M, N, d);
    std::cout << "--GPU Scores(Tiled) Execution time: " << ms << "ms --\n";
    CUDA_CHECK(cudaGetLastError());
}