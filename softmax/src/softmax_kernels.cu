#include "../include/softmax_kernels.cuh"
#include "../include/cuda_utils.cuh"

#include <cmath>


// one thread per row implementation
__global__ void compute_softmax_kernel1(float *scores, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M)
    {
        // finding max in this row
        float row_max = -INFINITY;
        for (int col = 0; col < N; col++)
        {
            float val = scores[row * N + col];
            if (val > row_max)
                row_max = val;
        }
        // computing exp(score - max) and sum
        float norm = 0.0f;
        for (int col = 0; col < N; col++)
        {
            float e = expf(scores[row * N + col] - row_max);
            scores[row * N + col] = e;
            norm = norm + e;
        }
        for (int col = 0; col < N; col++)
        {
            scores[row * N + col] = scores[row * N + col] / norm;
        }
    }
}

//one block per row and reduction
__global__ void compute_softmax_kernel2(float *scores, int M, int N)
{
    int row = blockIdx.x;  // row 0 for example has [1,2,3,4,3,5,6,4,6,4,7,8,7]
    int tid = threadIdx.x; // say thread id 0 is taken and the block corresponding to thread id 0 is 0

    extern __shared__ float smem[]; // dynamic shared mem only for the block(per block) based on no of threads per block
    if (row >= M)
        return;

    // step 1 local max per thread
    float local_max = -INFINITY;
    for (int col = tid; col < N; col = col + blockDim.x)
    {
        int id = row * N + col;
        float val = scores[id];
        local_max = fmax(local_max, val);
    }
    smem[tid] = local_max;
    __syncthreads();

    // reduction of local_max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] = fmax(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // compute exp(score-max) and local sum per thread
    float local_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x)
    {
        int id = row * N + col;
        float e = expf(scores[id] - row_max);
        scores[id] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();

    // reduction of local_sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = smem[0];
    __syncthreads();

    // step 3: normalize
    for (int col = tid; col < N; col += blockDim.x)
    {
        int id = row * N + col;
        scores[id] = scores[id] / row_sum;
    }
}

//online softmax
__global__ void compute_softmax_online_kernel3(float* scores, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= M)
        return;
    float m = -INFINITY;
    float l = 0.0f;
    // pass 1 online softmax
    for (int col = 0; col < N; col++)
    {
        int id = row * N + col;
        float x = scores[id];
        float new_max = fmaxf(x, m);
        l = l * expf(m - new_max) + expf(x - new_max);
        m = new_max;
    }
    // pass 2 do normalization
    for (int col = 0; col < N; col++)
    {
        int id = row * N + col;
        float x = scores[id];
        scores[id] = expf(x - m) / l;
    }
}

void launch_softmax_naive(float* d_scores, int M, int N) {
    dim3 block(256);
    dim3 grid((M + block.x - 1) / block.x);

    compute_softmax_kernel1<<<grid, block>>>(d_scores, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_shared(float* d_scores, int M, int N) {
    dim3 block(256);
    dim3 grid(M);

    size_t shared_mem = block.x * sizeof(float);

    compute_softmax_kernel2<<<grid, block, shared_mem>>>(d_scores, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_online(float* d_scores, int M, int N) {
    dim3 block(256);
    dim3 grid((M + block.x - 1) / block.x);

    compute_softmax_online_kernel3<<<grid, block>>>(d_scores, M, N);
    CUDA_CHECK(cudaGetLastError());
}