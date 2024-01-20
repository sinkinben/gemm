# $@ denote the target name
# $+ denote all the requirements


INCLUDE=../include/
FLGAS=-O3 -I $(INCLUDE) # -D OPEN_CHECKING=1
CC=g++ -std=c++17 $(FLGAS)

CPU_DIR=cpu-gemm

cpu_naive: cpu_naive.cpp main.cpp
	$(CC) $+ -o $@

cpu_opt_loop: cpu_opt_loop.cpp main.cpp
	$(CC) $+ -o $@

cpu_multi_threads: cpu_multi_threads.cpp main.cpp
	$(CC) $+ -o $@ -lpthread

cpu_simd: cpu_simd.cpp main.cpp
	$(CC) -mavx $+ -o $@ -lpthread

test: cpu_naive cpu_opt_loop cpu_multi_threads cpu_simd
	@echo -e 'Name\t\t    AvgTime(ms)\t\tAvgCycles'
	@./cpu_naive
	@./cpu_opt_loop
	@./cpu_multi_threads
	@./cpu_simd

clean:
	rm cpu_naive cpu_opt_loop cpu_multi_threads cpu_simd

