#include "cuda_matrix.h"
#include <stdio.h>

// Call by host, parameter must be passed by value (can not be passed by reference)
__global__ void DeviceMatrixMul(CudaMatrix A, CudaMatrix B, CudaMatrix C)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.x * CUDA_BLOCK_SIZE + tx;
    int col = blockIdx.y * CUDA_BLOCK_SIZE + ty;

    /* result of C[i, j] */
    float res = 0;

    /* shared memory within block */
    __shared__ float sharedA[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
    __shared__ float sharedB[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

    for (int k = 0; k < A.cols; k += CUDA_BLOCK_SIZE)
    {
        sharedA[tx][ty] = (row < A.rows && k + ty < A.cols) ? A.Get(row, k + ty) : 0;
        sharedB[tx][ty] = (k + tx < B.rows && col < B.cols) ? B.Get(k + tx, col) : 0;
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < CUDA_BLOCK_SIZE; ++j)
            res += sharedA[tx][j] * sharedB[j][ty];
        __syncthreads();
    }
    if (row < C.rows && col < C.cols)
        C.Get(row, col) = res;
}

void CudaMatrixMul(CudaMatrix a, CudaMatrix b, CudaMatrix c)
{
    /* One thread compute one element C[i, j].
     * Max number of threads per block is 1024.
     */
    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim(c.rows / blockDim.x + (c.rows % blockDim.x != 0),
                 c.cols / blockDim.y + (c.cols % blockDim.y != 0));
    DeviceMatrixMul<<<gridDim, blockDim>>>(a, b, c);

    assert(cudaDeviceSynchronize() == cudaSuccess);
}
