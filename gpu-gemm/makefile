# $@ denote the target name
# $+ denote all the requirements



INCLUDE=../include/
FLGAS=-O3 -I $(INCLUDE) -D OPEN_CHECKING=1
CC=nvcc -std=c++14 $(FLGAS)

cuda_naive: cuda_naive.cu main.cpp
	$(CC) $+ -o $@

test: cuda_naive
	@echo -e 'Name\t\t    AvgTime(ms)\t\tAvgCycles'
	@./cuda_naive


clean:
	rm cuda_naive