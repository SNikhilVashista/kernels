#include "../include/cuda_utils.cuh"
#include "../include/scores_kernels.cuh"
#include "../include/softmax_kernels.cuh"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    int M = 32768;
    int N = 32768;
    int d = 64;

    std::cout << "M x N x d = "
              << M << " x " << N << " x " << d << std::endl;

    size_t querysize = (size_t)M * d;
    size_t ksize = (size_t)N * d;
    size_t vsize = (size_t)N * d;
    size_t osize = (size_t)M * d;
    size_t scores = (size_t)M * N;

    std::vector<float> h_query(querysize);
    std::vector<float> h_key(ksize);
    std::vector<float> h_value(vsize);
    std::vector<float> h_output(osize, 0.0f);

    srand(time(0));

    for (size_t i = 0; i < querysize; i++) h_query[i] = rand() % 10;
    for (size_t i = 0; i < ksize; i++) h_key[i] = rand() % 10;
    for (size_t i = 0; i < vsize; i++) h_value[i] = rand() % 10;

    float* d_query = nullptr;
    float* d_key = nullptr;
    float* d_value = nullptr;
    float* d_output = nullptr;
    float* d_scores = nullptr;

    CUDA_CHECK(cudaMalloc(&d_query, querysize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_key, ksize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_value, vsize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, osize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, scores * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), querysize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), ksize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), vsize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, h_output.data(), osize * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    launch_scores_tiled(d_query, d_key, d_scores, M, N, d);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "--GPU Scores Execution time: " << ms << "ms --\n";

    CUDA_CHECK(cudaEventRecord(start));
    launch_softmax_shared(d_scores, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "--GPU Softmax (Block per row + reduction) Execution time: " << ms << "ms --\n";

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_output));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}