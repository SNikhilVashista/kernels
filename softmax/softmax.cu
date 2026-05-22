#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
using namespace std;
#define CUDA_CHECK(call)                             \
    do                                               \
    {                                                \
        cudaError_t err = call;                      \
        if (err != cudaSuccess)                      \
        {                                            \
            cerr << "CUDA error at "                 \
                 << __FILE__ << " : "                \
                 << __LINE__ << " - "                \
                 << cudaGetErrorString(err) << endl; \
            exit(1);                                 \
        }                                            \
    } while (0)

__global__ void compute_scores_kernel(float *Q, float *K, float *scores, int M, int N, int d)
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

int main()
{
    //  attention(q,k,v) = softmax(qxk(t)/sqrt(d_k)*V)
    size_t M = 30000, N = 30000, d = 64; // number of query rows, key/value rows and head dimension
    size_t querysize = M * d;
    size_t ksize = N * d;
    size_t vsize = N * d;
    size_t osize = M * d;
    size_t scores = M * N; // N comes from ksize as it is N*d, it will become transpose for score computation.
    //========Host pointers===
    vector<float> h_query(querysize);
    vector<float> h_key(ksize);
    vector<float> h_value(vsize);
    vector<float> h_output(osize);
    vector<float> h_scores(scores);
    //========Device Pointers
    float *d_query;
    float *d_key;
    float *d_values;
    float *d_output;
    float *d_scores;

    srand(time(0));
    for (size_t i = 0; i < querysize; i++)
    {
        h_query[i] = static_cast<float>(rand() % 10);
    }
    for (size_t i = 0; i < ksize; i++)
    {
        h_key[i] = static_cast<float>(rand() % 10);
    }
    for (size_t i = 0; i < vsize; i++)
    {
        h_value[i] = static_cast<float>(rand() % 10);
    }
    for (size_t i = 0; i < osize; i++)
    {
        h_output[i] = 0.0f;
    }
    for (size_t i = 0; i < scores; i++)
    {
        h_scores[i] = 0.0f;
    }
    CUDA_CHECK(cudaMalloc(&d_query, querysize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_key, ksize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values, vsize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, scores * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, osize * sizeof(float)));

    //==MEMCOPY==
    CUDA_CHECK(cudaMemcpy(d_query, &h_query[0], querysize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, &h_key[0], ksize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, &h_value[0], vsize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, &h_output[0], osize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadperblock(32, 32);
    dim3 numblocks(
        (N + threadperblock.x - 1) / threadperblock.x,
        (M + threadperblock.y - 1) / threadperblock.y);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /*
    Mental model:
    Scores row = Q row
    Scores column = K row
    Each score = dot product over d features
    */
    float ms = 0.0f;
    cudaEventRecord(start);
    compute_scores_kernel<<<numblocks, threadperblock>>>(d_query, d_key, d_scores, M, N, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cout << "--Avg GPU Scores Execution time: " << ms << "ms --\n";

    //===softmax-gpu-kernel--launch
    // dim3 softmaxthreads(1024);
    // dim3 softmaxBlocks((M+softmaxthreads.x-1)/softmaxthreads.x);
    // vector<float> h_scores_gpu(scores);
    // CUDA_CHECK(cudaMemcpy(&h_scores_gpu[0],d_scores,scores * sizeof(float),cudaMemcpyDeviceToHost));
    // cudaEventRecord(start);
    // compute_softmax_kernel1<<<softmaxBlocks,softmaxthreads>>>(d_scores,M,N);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&ms, start, stop);
    // cout<<"--GPU Softmax Execution time: "<< ms <<"ms --\n";
    // vector<float> h_softmax_gpu(scores);
    // CUDA_CHECK(cudaMemcpy(&h_softmax_gpu[0],d_scores,scores*sizeof(float),cudaMemcpyDeviceToHost));

    //==softmax-gpu-kernal2-launch
    dim3 softmaxthreadsperblock(256);
    dim3 softmaxblockdim(M);
    size_t sharedmemsize = softmaxthreadsperblock.x * sizeof(float);

    cudaEventRecord(start);
    compute_softmax_kernel2<<<softmaxblockdim, softmaxthreadsperblock, sharedmemsize>>>(
        d_scores, M, N);
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cout << "--GPU Softmax Kernel2 Execution time: " << ms << "ms --\n";

    vector<float> h_softmax_gpu(scores);

    CUDA_CHECK(cudaMemcpy(
        &h_softmax_gpu[0],
        d_scores,
        scores * sizeof(float),
        cudaMemcpyDeviceToHost));
    cudaFree(d_scores);
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_values);
    cudaFree(d_output);
    auto cstart = chrono::high_resolution_clock::now();
    // q*kt computation
    for (size_t q_row = 0; q_row < M; q_row++)
    {
        for (size_t k_row = 0; k_row < N; k_row++)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < d; i++)
            {
                sum = sum + h_query[q_row * d + i] * h_key[k_row * d + i];
            }
            h_scores[q_row * N + k_row] = sum / sqrt((float)d);
        }
    }
    auto cend = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = cend - cstart;
    cout << "--CPU Scores Time = " << elapsed.count() << "ms --\n";

    cstart = chrono::high_resolution_clock::now();
    // attention weight calculation on cpu
    for (size_t row = 0; row < M; row++)
    {
        float row_max = -INFINITY;
        // find row max for this row
        for (size_t col = 0; col < N; col++)
        {
            size_t i = row * N + col;
            row_max = max(row_max, h_scores[i]);
        }
        // calc exp sum
        float norm = 0.0f;
        for (size_t col = 0; col < N; col++)
        {
            size_t i = row * N + col;
            float e = expf(h_scores[i] - row_max);
            // inplace storing of e and accumulating norm
            h_scores[i] = e;
            norm = norm + e;
        }
        // normalize
        for (size_t col = 0; col < N; col++)
        {
            size_t i = row * N + col;
            h_scores[i] = h_scores[i] / norm;
        }
    }
    cend = chrono::high_resolution_clock::now();
    elapsed = cend - cstart;
    cout << "--CPU Softmax Time = " << elapsed.count() << "ms --\n";

    // output matrix computation
    //  for (size_t q_row = 0; q_row < M; q_row++){
    //      for(size_t col=0;col<d;col++){
    //          float sum = 0.0f;
    //          for(size_t v_row=0;v_row<N;v_row++){
    //              float attentionweight=h_scores[q_row*N+v_row];
    //              float v_value = h_value[v_row*d+col];
    //              sum = sum+attentionweight*v_value;
    //          }
    //          h_output[q_row*d+col]=sum;
    //      }
    //  }
}

/*
Benchmark: M = 30000, N = 30000, d = 64
GPU: NVIDIA GeForce GTX 1650 Max-Q, sm_75

Naive softmax kernel 1:
- Scores kernel: one thread computes one Scores[row][col]
- Softmax kernel: one thread computes one full row

Output:
--Avg GPU Scores Execution time: 4933.79ms --
--GPU Softmax Execution time: 4334.82ms --
--CPU Scores Time = 78359.7ms --
--CPU Softmax Time = 17781.5ms --

Optimized softmax kernel 2:
- Scores kernel: one thread computes one Scores[row][col]
- Softmax kernel: one block computes one row
- 256 threads per block cooperate using shared memory reduction

Output:
--Avg GPU Scores Execution time: 4914.19ms --
--GPU Softmax Kernel2 Execution time: 201.03ms --
--CPU Scores Time = 45228.3ms --
--CPU Softmax Time = 12342.5ms --

Softmax improvement:
4334.82 ms -> 201.03 ms
~21.56x faster
*/