#include <iostream>
#include <cmath>
#include <config.h>
#include <matrix.h>
#include <timer.h>
#include <cpu_info.h>
#include <assert.h>

extern void MatrixMul(Matrix &, Matrix &, Matrix &);

void CheckCorrect(Matrix &A, Matrix &B, Matrix &C)
{
    for (int i = 0; i < C.rows; ++i)
    {
        for (int j = 0; j < C.cols; ++j)
        {
            float expect = 0;
            for (int idx = 0; idx < A.cols; ++idx)
                expect += A.Get(i, idx) * B.Get(idx, j);
            assert(fabs(C.Get(i, j) - expect) < 1e-4);
        }
    }
}

void RunTest(int m, int n, int k, char *method_name)
{
    
    Matrix A(m, n), B(n, k), C(m, k);
    
    A.Randomize(), B.Randomize();
    Timer timer;
    uint64_t cycle_start = 0, total_cycles = 0;
    double total_time = 0;
    
    for (int i = 0; i < TEST_LOOP; ++i)
    {
        C.Fill(0);
        timer.reset();

        cycle_start = CpuInfo::GetCpuCycle();
        MatrixMul(A, B, C);
        total_cycles += CpuInfo::GetCpuCycle() - cycle_start;
        total_time += timer.elapsed_nano(); // ns
    }

    PrintResult(method_name, (total_time / 1e6) / TEST_LOOP, (double)total_cycles / TEST_LOOP);

#ifdef OPEN_CHECKING
    CheckCorrect(A, B, C);
#endif
}

int main(int argc, char *argv[])
{
    std::ios::sync_with_stdio(false);
    RunTest(MAT_SIZE, MAT_SIZE, MAT_SIZE, argv[0] + 2);
}