#pragma once

#include <cstddef>

extern "C" {
    void launchAddKernel(const float* A, const float* B, float* C, size_t N);
    void launchMatmulKernel(const float* A, const float* B, float* C, size_t M, size_t K, size_t N);
    void launchTiledMatmulKernel(const float* A, const float* B, float* C, size_t M, size_t K, size_t N);

    // memory management (cpp -> cuda bridge)
    float* allocateVram(size_t bytes);
    void freeVram(float* ptr);
    void copyMemory(float* dst, const float* src, size_t bytes, bool toGpu);
    void fillZerosVram(float* ptr, size_t bytes);
}
