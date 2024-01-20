## GEMM Step by Step

**GEMM on CPU**
- `cpu_naive`: The most naive triple for loops method.
- `cpu_opt_loop`: Optimize the order of for-loop, let 3 matrices be access as row-oriented.
- `cpu_multi_threads`: Divide result matrix as some parts, one thread to compute the results of one part.
- `cpu_simd`: Based on multi-threads, one thread will execute in one core, leverage SIMD to optimize the most internal for-loop.

**Usage**
Run `make cpu_xxx` or `make cuda_xxx` to execute one program. Run `make test` to execute all programs.
