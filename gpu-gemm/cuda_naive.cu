#include "cuda_matrix.h"
#include <stdio.h>

// Device function can not pass parameter by references, since the data of object 'a, b, c' is stored in host memory
// the a.Get() will access the pointer of 'a' if pass by reference
__global__ void DeviceMatrixMul(CudaMatrix a, CudaMatrix b, CudaMatrix c)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < c.rows && col < c.cols)
    {
        float res = 0;
        for (int j = 0; j < a.cols; ++j)
            res += a.Get(row, j) * b.Get(j, col);
        c.Get(row, col) = res;
    }
}

void CudaMatrixMul(CudaMatrix a, CudaMatrix b, CudaMatrix c)
{
    /* check validity of input matrice */
    // assert(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols);

    /* One thread compute one element C[i, j].
     * Max number of threads per block is 1024.
     */
    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim(c.rows / blockDim.x + (c.rows % blockDim.x != 0),
                 c.cols / blockDim.y + (c.cols % blockDim.y != 0));
    DeviceMatrixMul<<<gridDim, blockDim>>>(a, b, c);

    assert(cudaDeviceSynchronize() == cudaSuccess);
}
