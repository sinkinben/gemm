# $@ denote the target name
# $+ denote all the requirements


FLGAS=-O3 -Iinclude/
CC=g++ -std=c++17 $(FLGAS)
NVCC=nvcc $(FLGAS) # -D CUDA=1

CPU_DIR=cpu-gemm

cpu_naive: $(CPU_DIR)/cpu_naive.cpp main.cpp
	$(CC) $+ -o $@

cpu_opt_loop: $(CPU_DIR)/cpu_opt_loop.cpp main.cpp
	$(CC) $+ -o $@

test: cpu_naive cpu_opt_loop
	@echo -e 'Name\t\t    AvgTime(ms)\t\tAvgCycles'
	@./cpu_naive
	@./cpu_opt_loop


clean:
	rm cpu_naive cpu_opt_loop temp.exe

tmp:
	$(CC) $(FLGAS) temp.cpp -o temp.exe
	./temp.exe
