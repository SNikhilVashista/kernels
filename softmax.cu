#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<chrono>
/*
Pass 1 - Calculation of the maximum: The whole input row is first traversed from left (index = 0) to right (index = N - 1) to find the maximum value  xmax.

Pass 2 - Calculation of the norm: The whole input row is traversed from left to right again, but this time the normalization factor is computed using the 
xmax value from the first pass, for each element.

Pass 3 - Softmax computation: The whole input row is traversed again from left to right and for each element the exponential of 
(x−xmax)  is divided by the norm calculated in the second pass.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at line " << __LINE__ << endl;\
            exit(1);                                          \
        }                                                     \
    } while (0)

__global__ void softmax_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int M,
    int N,
    int d
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    float* scores = shared;          // size N
    float* partial = shared + N;     // size blockDim.x
    float scale = rsqrtf((float)d);
    // Compute scores[row, j] = Q[row] dot K[j]
    for (int j = tid; j < N; j += blockDim.x) {
        float dot = 0.0f;

        for (int k = 0; k < d; k++) {
            dot += Q[row * d + k] * K[j * d + k];
        }

        scores[j] = dot * scale;
    }
    __syncthreads();
    // pass 1. Find max score for numerical stability
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, scores[j]);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    float max_score = partial[0];
    // pass 2. Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        scores[j] = expf(scores[j] - max_score);
        local_sum += scores[j];
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = partial[0];
    // pass 3. Compute output O[row, col]
    for (int col = tid; col < d; col += blockDim.x) {
        float out = 0.0f;
        for (int j = 0; j < N; j++) {
            float weight = scores[j] / sum_exp;
            out += weight * V[j * d + col];
        }
        O[row * d + col] = out;
    }
}
void softmax_attention_cpu(
    const vector<float>& Q,
    const vector<float>& K,
    const vector<float>& V,
    vector<float>& O,
    int M,
    int N,
    int d
) {
    float scale = 1.0f / sqrt((float)d);
    vector<float> scores(N);
    for (int i = 0; i < M; i++) {
        float max_score = -INFINITY;
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += Q[i * d + k] * K[j * d + k];
            }
            scores[j] = dot * scale;
            max_score = max(max_score, scores[j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < N; j++) {
            scores[j] = exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int col = 0; col < d; col++) {
            float out = 0.0f;
            for (int j = 0; j < N; j++) {
                float weight = scores[j] / sum_exp;
                out += weight * V[j * d + col];
            }
            O[i * d + col] = out;
        }
    }
}
int main() {
    int M = 128;   // number of query tokens
    int N = 128;   // number of key/value tokens
    int d = 64;    // head dimension
    printf("Running CUDA softmax attention benchmark on MxN = (%d,% d)\n",M,N);
    size_t q_size = M * d * sizeof(float);
    size_t k_size = N * d * sizeof(float);
    size_t v_size = N * d * sizeof(float);
    size_t o_size = M * d * sizeof(float);
    vector<float> h_Q(M * d, 1.0f);
    vector<float> h_K(N * d, 1.0f);
    vector<float> h_V(N * d, 1.0f);
    vector<float> h_O(M * d, 0.0f);
    vector<float> h_O_cpu(M * d, 0.0f);
    float *d_Q, *d_K, *d_V, *d_O;
    float ms=0;
    srand(42);
    for (int i = 0; i < M * d; i++) {
    h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    for (int i = 0; i < N * d; i++) {
    h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    cudaEventRecord(start);
    CHECK_CUDA(cudaMalloc(&d_Q, q_size));
    CHECK_CUDA(cudaMalloc(&d_K, k_size));
    CHECK_CUDA(cudaMalloc(&d_V, v_size));
    CHECK_CUDA(cudaMalloc(&d_O, o_size));
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventSynchronize(stop);
    cout<<"--GPU allocation time: "<< ms <<"ms --\n";
    cudaEventRecord(start);
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), k_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), v_size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cout<<"--Host 2 Device transfer time: "<< ms <<"ms --\n";
    int threads = 256;
    int blocks = M;
    size_t shared_mem = (N + threads) * sizeof(float);
    cudaEventRecord(start);
    softmax_attention_kernel<<<blocks, threads, shared_mem>>>(
        d_Q, d_K, d_V, d_O, M, N, d
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cout<<"--Kernel Execution time: "<< ms <<"ms --\n";
    cudaEventRecord(start);
    CHECK_CUDA(cudaMemcpy(h_O.data(), d_O, o_size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cout<<"--Device 2 Host transfer time: "<< ms <<"ms --\n";
    auto cstart = chrono::high_resolution_clock::now();
    softmax_attention_cpu(h_Q, h_K, h_V, h_O_cpu, M, N, d);
    auto cend = chrono::high_resolution_clock::now();
    chrono::duration<double,milli> elapsed = cend-cstart;
    cout<<"--CPU Time = "<<elapsed.count()<<"ms --\n";
    float max_error = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < M * d; i++) {
        float error = fabs(h_O[i] - h_O_cpu[i]);
        max_error = max(max_error, error);
        if (error > 1e-4) {
            mismatch_count++;

            if (mismatch_count <= 10) {
                cout << "Mismatch at index " << i
                        << " GPU = " << h_O[i]
                        << " CPU = " << h_O_cpu[i]
                        << " error = " << error
                        << endl;
            }
        }
    }

    cout << "Max error = " << max_error << endl;
    cout << "Mismatch count = " << mismatch_count << endl;
    if (mismatch_count == 0) {
        cout << "Test PASSED. All outputs match." << endl;
    } else {
        cout << "Test FAILED." << endl;
    }
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    return 0;
}