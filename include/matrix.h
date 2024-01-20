#pragma once
#include <cstdlib>
#include <algorithm>
#include <cassert>
constexpr size_t kAlignment = 64;
class Matrix
{
public:
    int rows, cols;
    float *elements;
    Matrix(int m, int n) : rows(m), cols(n), elements(new float[m * n])
    {
    }

    virtual ~Matrix()
    {
        if (elements != nullptr)
            delete[] elements;
    }

    void Randomize()
    {
        size_t size = GetSize();
        for (size_t i = 0; i < size; ++i)
        {
            elements[i] = (rand() % 10) * 0.50f;
        }
    }

    void Fill(float val = 0)
    {
        std::fill(elements, elements + GetSize(), val);
    }

    inline size_t GetSize() const { return (size_t)rows * cols; }

    inline float &Get(int i, int j) { return elements[i * cols + j]; }
};