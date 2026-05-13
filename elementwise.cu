#include<iostream>
#include<chrono>

__global__ void addKernel(float* A,float* B, float* C, int size){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size)
        C[idx] = A[idx]+B[idx];
}

int main(){
    int sizes[] = {1000000, 10000000, 50000000};
    int tpbs[] = {128, 256, 512};   
    for (int s = 0; s < 3; s++) {
    for (int t = 0; t < 3; t++) { 
        int size = sizes[s];
        int tpb = tpbs[t];
        float* A = new float[size];
        float* B = new float[size];
        float* C = new float[size];
        for (int i = 0; i < size; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
        }
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++) {
            C[i] = A[i] + B[i];
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;   
        std::cout << "CPU: " << cpu_time.count() << " ms\n";
        float *d_A,*d_B,*d_C;
        cudaMalloc(&d_A,size * sizeof(float));
        cudaMalloc(&d_B,size * sizeof(float));
        cudaMalloc(&d_C,size * sizeof(float));
        auto h2d_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
        auto h2d_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> h2d_time = h2d_end - h2d_start;
        int bpg = (size + tpb - 1)/tpb;
        auto start = std::chrono::high_resolution_clock::now();
        addKernel<<<bpg,tpb>>>(d_A,d_B,d_C,size);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration = end-start;
        auto d2h_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(C,d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
        auto d2h_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> d2h_time = d2h_end - d2h_start;
        std::cout << "GPU size: " << size
          << " | tpb: " << tpb
          << " | bpg: " << bpg
          << " | H2D: " << h2d_time.count() << " ms"
          << " | Kernel: " << duration.count() << " ms"
          << " | D2H: " << d2h_time.count() << " ms"
          << " | Total: " << h2d_time.count() + duration.count() + d2h_time.count() << " ms"
          << " | C[last]: " << C[size - 1]
          << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] A;
        delete[] B;
        delete[] C;
        }
        }
    // std::cout << C[size - 1] << std::endl;
    return 0;
}

// CPU: 2.1571 ms
// GPU size: 1000000 | tpb: 128 | bpg: 7813 | H2D: 1.8738 ms | Kernel: 0.9675 ms | D2H: 0.9228 ms | Total: 3.7641 ms | C[last]: 2
// CPU: 2.3233 ms
// GPU size: 1000000 | tpb: 256 | bpg: 3907 | H2D: 1.8389 ms | Kernel: 0.2137 ms | D2H: 0.8098 ms | Total: 2.8624 ms | C[last]: 2
// CPU: 2.316 ms
// GPU size: 1000000 | tpb: 512 | bpg: 1954 | H2D: 1.8659 ms | Kernel: 0.252 ms | D2H: 0.8633 ms | Total: 2.9812 ms | C[last]: 2
// CPU: 24.8957 ms
// GPU size: 10000000 | tpb: 128 | bpg: 78125 | H2D: 20.6845 ms | Kernel: 0.9685 ms | D2H: 8.4961 ms | Total: 30.1491 ms | C[last]: 2
// CPU: 22.4798 ms
// GPU size: 10000000 | tpb: 256 | bpg: 39063 | H2D: 21.6442 ms | Kernel: 0.9252 ms | D2H: 9.1094 ms | Total: 31.6788 ms | C[last]: 2
// CPU: 23.3422 ms
// GPU size: 10000000 | tpb: 512 | bpg: 19532 | H2D: 26.0927 ms | Kernel: 0.9431 ms | D2H: 8.911 ms | Total: 35.9468 ms | C[last]: 2
// CPU: 115.952 ms
// GPU size: 50000000 | tpb: 128 | bpg: 390625 | H2D: 87.3015 ms | Kernel: 4.1917 ms | D2H: 33.1651 ms | Total: 124.658 ms | C[last]: 2
// CPU: 105.852 ms
// GPU size: 50000000 | tpb: 256 | bpg: 195313 | H2D: 89.3401 ms | Kernel: 4.1636 ms | D2H: 35.5497 ms | Total: 129.053 ms | C[last]: 2
// CPU: 103.308 ms
// GPU size: 50000000 | tpb: 512 | bpg: 97657 | H2D: 85.7883 ms | Kernel: 4.2017 ms | D2H: 32.3847 ms | Total: 122.375 ms | C[last]: 2