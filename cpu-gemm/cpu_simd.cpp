/* GEMM using mutlti-threads and SIMD 
 * 1. Add '-mavx' flags to g++ compiler when compiling cpu_unroll.cpp
 * 2. Unroll the for-loop, to tell the g++ compiler, the scope of 'for' can be vectorized
 */

#include <matrix.h>
#include <thread>

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
