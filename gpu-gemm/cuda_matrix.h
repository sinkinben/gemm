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
        assert(cudaMalloc(&elements, (size_t)m * n * sizeof(float)) == cudaSuccess);
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

    size_t GetSize() const { return size_t(rows) * cols; }

    __device__ float &Get(int i, int j) const
    {
        return elements[i * cols + j];
    }
};
