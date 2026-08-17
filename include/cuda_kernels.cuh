#pragma once

#include <cstddef>

extern "C" {
    void launchAddKernel(const double* A, const double* B, double* C, size_t N);
    void launchMatmulKernel(const double* A, const double* B, double* C, size_t M, size_t K, size_t N);
    void launchTiledMatmulKernel(const double* A, const double* B, double* C, size_t M, size_t K, size_t N);

    // memory management (cpp -> cuda bridge)
    double* allocateVram(size_t bytes);
    void freeVram(double* ptr);
    void copyMemory(double* dst, const double* src, size_t bytes, bool toGpu);
    void fillZerosVram(double* ptr, size_t bytes);
}
