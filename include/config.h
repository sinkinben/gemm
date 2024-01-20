#pragma once
#include <stdio.h>

constexpr int TEST_LOOP = 1;   // Test case loop
constexpr int MAT_SIZE = 1024; // Rows, cols of each matrix, it must be times of 8

#define PrintResult(name, avgTime, flops) \
    printf("%-20s%-20lf%-20lf\n", (name), (avgTime), (flops))

