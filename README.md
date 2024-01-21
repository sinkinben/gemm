## GEMM Step by Step

**GEMM on CPU**

Recommend to read the code in this sequence order.
- `cpu_naive`: The most naive triple for loops method.
- `cpu_opt_loop`: Optimize the order of for-loop, let 3 matrices be access as row-oriented, more cache-friendly.
- `cpu_multi_threads`: Divide result matrix as some parts, one thread to compute the results of one part.
- `cpu_simd`: Based on multi-threads, one thread will execute in one core, leverage SIMD to optimize the most internal for-loop.

**GEMM on GPU**

- `cuda_naive`: The most naive method to compute GEMM with CUDA. One thread to compute one cell.
- `cuda_shared_mem`: Divide the matrix into some tiles, leverage the shared memory within each block to reduce the time of memory access. Refer to [CUDA Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory).

**Usage**

Run `make cpu_xxx` or `make cuda_xxx` to execute one program. Run `make test` to execute all programs.

Final result on my machine:
```text
matrix size = 1024 x 1024
Name                AvgTime(ms)         AvgCpuCycles
cpu_naive           2906.039965         6695144468.000000
cpu_opt_loop        208.853657          481170332.000000
cpu_multi_threads   89.721127           206703648.000000
cpu_simd            76.574855           176416902.000000
cuda_naive          203.609718          -
cuda_shared_mem     70.912909           -
```
