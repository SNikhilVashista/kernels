# CUDA Kernel Implementations

This repository contains basic CUDA and C++ implementations of common GPU operations, including convolution, elementwise operations, and softmax.

## Project Structure

```text
.
├── convolution
│   ├── conv2x2.cu
│   └── convolution1x1.cpp
├── elementwise
│   ├── elementwise.cpp
│   └── elementwise.cu
├── softmax
│   └── softmax.cu
└── .gitignore
```

## Implemented Operations

### Convolution

The `convolution` folder contains convolution implementations.

- `conv2x2.cu` — CUDA implementation of a 2x2 convolution kernel
- `convolution1x1.cpp` — C++ implementation of 1x1 convolution

### Elementwise Operations

The `elementwise` folder contains CPU and CUDA implementations of elementwise operations.

- `elementwise.cpp` — CPU implementation
- `elementwise.cu` — CUDA implementation

### Softmax

The `softmax` folder contains a CUDA implementation of softmax.

- `softmax.cu` — CUDA softmax implementation

## Requirements

- NVIDIA GPU
- CUDA Toolkit
- C++ compiler
- NVIDIA CUDA compiler `nvcc`

## How to Compile

Compile the softmax CUDA program:

```bash
nvcc softmax/softmax.cu -o softmax.exe
```

Compile the elementwise CUDA program:

```bash
nvcc elementwise/elementwise.cu -o elementwise.exe
```

Compile the convolution CUDA program:

```bash
nvcc convolution/conv2x2.cu -o conv2x2.exe
```

Compile the C++ 1x1 convolution program:

```bash
g++ convolution/convolution1x1.cpp -o convolution1x1.exe
```

On Windows, if you are using MSVC instead of `g++`, compile with:

```cmd
cl convolution\convolution1x1.cpp
```

## How to Run

On Windows Command Prompt:

```cmd
softmax.exe
elementwise.exe
conv2x2.exe
convolution1x1.exe
```

On Linux or Git Bash:

```bash
./softmax.exe
./elementwise.exe
./conv2x2.exe
./convolution1x1.exe
```


## Notes

This repository is mainly for learning and practice. The implementations are simple versions of operations commonly used in deep learning workloads.