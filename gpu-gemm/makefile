# $@ denote the target name
# $+ denote all the requirements



INCLUDE=../include/
FLGAS=-O3 -I $(INCLUDE) -D OPEN_CHECKING=1
CC=nvcc -std=c++14 $(FLGAS)

cuda_naive: cuda_naive.cu main.cpp
	$(CC) $+ -o $@

cuda_shared_mem: cuda_shared_mem.cu main.cpp
	$(CC) $+ -o $@

test: cuda_naive cuda_shared_mem
	@echo -e 'Name\t\t    AvgTime(ms)\t\tAvgCycles'
	@./cuda_naive
	@./cuda_shared_mem

tmp: temp.cu
	$(CC) $+ -o $@

clean:
	rm cuda_naive cuda_shared_mem