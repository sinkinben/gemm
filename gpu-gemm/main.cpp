#include <config.h>
#include <matrix.h>
#include <timer.h>
#include "cuda_matrix.h"

#ifdef OPEN_CHECKING
#include "cpu_bench.h"
#endif

extern void CudaMatrixMul(CudaMatrix a, CudaMatrix b, CudaMatrix c);

void RunTest(int m, int n, int k, const char *method)
{
    CudaMatrix A(m, n), B(n, k), C(m, k); // Matrix in device memory

    Matrix a(m, n), b(n, k); // Matrix in host memory
    {
        a.Randomize(), b.Randomize();
        A.FillData(a.elements, m * n);
        B.FillData(b.elements, n * k);
        assert(cudaDeviceSynchronize() == cudaSuccess); // wait to finish copying
    }

    Timer timer;
    uint64_t cycle_start = 0;
    double total_time = 0;

    for (int i = 0; i < TEST_LOOP; ++i)
    {
        timer.reset();
        CudaMatrixMul(A, B, C);
        total_time += timer.elapsed_nano(); // ns
    }

    PrintResult(method, (total_time / 1e6) / TEST_LOOP, 0.0 / TEST_LOOP);

#ifdef OPEN_CHECKING
    Matrix cpu_result(n, k);
    CpuBench::CpuMatrixMul(a, b, cpu_result);
    CpuBench::CheckCorrect(cpu_result, C);
#endif

    // Destroy memory in device by hand
    A.Destroy(), B.Destroy(), C.Destroy();
}

int main(int argc, char *argv[])
{
    RunTest(MAT_SIZE, MAT_SIZE, MAT_SIZE, argv[0] + 2);
}
