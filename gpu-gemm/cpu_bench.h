#pragma once
#include <matrix.h>
#include <math.h>
#include <thread>
#include <vector>
#include "cuda_matrix.h"

namespace CpuSimdGemm
{
    // Copied from cpu_simd.cpp
    void Worker(Matrix &A, Matrix &B, Matrix &C, int x1, int y1, int x2, int y2)
    {
        int n = A.cols;
        float *pa, *pb, *pc;
        float a = 0;
        for (int i = x1; i < x2; ++i)
        {
            for (int idx = 0; idx < n; ++idx)
            {
                a = A.Get(i, idx);
                // #pragma unroll
                for (int j = y1; j < y2; j += 8)
                {
                    // C.Get(i, j) += A.Get(i, idx) * B.Get(idx, j);
                    pc = &C.Get(i, j), pb = &B.Get(idx, j);
                    pc[0] += a * pb[0];
                    pc[1] += a * pb[1];
                    pc[2] += a * pb[2];
                    pc[3] += a * pb[3];
                    pc[4] += a * pb[4];
                    pc[5] += a * pb[5];
                    pc[6] += a * pb[6];
                    pc[7] += a * pb[7];
                }
            }
        }
    }

    void MatrixMul(Matrix &A, Matrix &B, Matrix &C)
    {
        constexpr int d = 2;
        int m = C.rows, k = C.cols;
        int row_gap = m / d, col_gap = k / d;
        std::vector<std::thread> pool;
        for (int i = 0; i < m; i += row_gap)
        {
            for (int j = 0; j < k; j += col_gap)
                pool.emplace_back(std::thread(Worker, std::ref(A), std::ref(B), std::ref(C), i, j, i + row_gap, j + col_gap));
        }
        for (auto &th : pool)
            th.join();
    }
};

namespace CpuBench
{
    
    static void CpuMatrixMul(Matrix &A, Matrix &B, Matrix &C)
    {
        CpuSimdGemm::MatrixMul(A, B, C);
    }
    static void CheckCorrect(Matrix &cpu_matrix, CudaMatrix &cuda_matrix)
    {
        size_t n_floats = cuda_matrix.GetSize();
        assert(cpu_matrix.GetSize() == n_floats);

        float *host_memory = new float[n_floats];
        float *expect = cpu_matrix.elements;
        assert(cudaMemcpy(host_memory, cuda_matrix.elements, n_floats * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(cudaDeviceSynchronize() == cudaSuccess);

        for (size_t i = 0; i < n_floats; ++i)
        {
            // std::printf("i = %d | expect = %f, but got %f, diff = %f\n", i, expect[i], host_memory[i], fabs(expect[i] - host_memory[i]));
            assert(fabs(expect[i] - host_memory[i]) < 1e-4);
        }
        delete[] host_memory;
    }
}
