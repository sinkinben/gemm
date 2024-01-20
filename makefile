# $@ denote the target name
# $+ denote all the requirements


FLGAS=-O3 -Iinclude/ -D OPEN_CHECKING=1
CC=g++ -std=c++17 $(FLGAS)
NVCC=nvcc $(FLGAS) # -D CUDA=1

CPU_DIR=cpu-gemm

cpu_naive: $(CPU_DIR)/cpu_naive.cpp main.cpp
	$(CC) $+ -o $@

cpu_opt_loop: $(CPU_DIR)/cpu_opt_loop.cpp main.cpp
	$(CC) $+ -o $@

cpu_multi_threads: $(CPU_DIR)/cpu_multi_threads.cpp main.cpp
	$(CC) $+ -o $@ -lpthread

test: clean cpu_naive cpu_opt_loop cpu_multi_threads
	@echo -e 'Name\t\t    AvgTime(ms)\t\tAvgCycles'
	@./cpu_naive
	@./cpu_opt_loop
	@./cpu_multi_threads


clean:
	rm cpu_naive cpu_opt_loop cpu_multi_threads temp.exe

tmp:
	$(CC) $(FLGAS) temp.cpp -o temp.exe
	./temp.exe
