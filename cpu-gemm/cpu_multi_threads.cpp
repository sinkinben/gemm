#include <matrix.h>
#include <thread>

// range of C(x1, y1) -> C(x2, y2), is computed by current thread
void Worker(Matrix &A, Matrix &B, Matrix &C, int x1, int y1, int x2, int y2)
{
    int n = A.cols;
    for (int i = x1; i < x2; ++i)
    {
        for (int idx = 0; idx < n; ++idx)
        {
            for (int j = y1; j < y2; ++j)
                C.Get(i, j) += A.Get(i, idx) * B.Get(idx, j);
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
