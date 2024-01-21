#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <assert.h>

constexpr int CUDA_BLOCK_SIZE = 32;

class CudaMatrix
{
public:
    int rows, cols;
    float *elements;

    CudaMatrix(int m, int n) : rows(m), cols(n)
    {
        size_t total_bytes = (size_t)m * n * sizeof(float);
        assert(cudaMalloc(&elements, total_bytes) == cudaSuccess);
        assert(cudaMemset(elements, 0, total_bytes) == cudaSuccess);
    }

    ~CudaMatrix()
    {
    }

    void Destroy()
    {
        assert(cudaFree(elements) == cudaSuccess);
    }

    void FillData(float *host_memory, size_t n_floats)
    {
        assert(cudaMemcpy(elements, host_memory, n_floats * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    void FillZero()
    {
        assert(cudaMemset(elements, 0, sizeof(float) * GetSize()) == cudaSuccess);
    }

    size_t GetSize() const { return size_t(rows) * cols; }

    __device__ float &Get(int i, int j) const
    {
        return elements[i * cols + j];
    }
};
