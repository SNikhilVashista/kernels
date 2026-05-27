#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                       \
    do                                                         \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess)                                \
        {                                                      \
            std::cerr << "CUDA error at " << __FILE__          \
                      << " : " << __LINE__ << " - "            \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(1);                                      \
        }                                                      \
    } while (0)