#include <benchmark/benchmark.h>
#include "Tensor.hpp"
#include "Arena.hpp" 
#include <vector>

static void BM_NaiveMatMul(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<double> dataA(size * size, 1.0);
    std::vector<double> dataB(size * size, 1.0);
    
    // Allocate A and B in paramArena (isParam = true)
    Tensor A(dataA, {size, size}, {size, 1}, true);
    Tensor B(dataB, {size, size}, {size, 1}, true);

    for (auto _ : state) {
        Tensor C = A * B; // Allocates C in globalArena
        benchmark::DoNotOptimize(C); 
        globalArena.reset(); 
    }
}

static void BM_TensorAdd(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<double> dataA(size * size, 1.0);
    std::vector<double> dataB(size * size, 2.0);
    
    Tensor A(dataA, {size, size}, {size, 1}, true);
    Tensor B(dataB, {size, size}, {size, 1}, true);

    for (auto _ : state) {
        Tensor C = A + B; 
        benchmark::DoNotOptimize(C); 
        globalArena.reset(); 
    }
}

static void BM_TiledMatMul(benchmark::State& state) {
    int size = state.range(0);
    
    // 1. Create tensors DIRECTLY on the GPU in paramArena (isParam = true).
    // This protects them from the globalArena reset in the loop.
    // They initialize to 0.0, which perfectly simulates math load.
    Tensor A_gpu({size, size}, true, Device::CUDA);
    Tensor B_gpu({size, size}, true, Device::CUDA);
    
    for (auto _ : state) {
        // 2. Timer starts: Multiply pure VRAM pointers (No memory allocation tax!)
        // Note: Our C++ wrapper inside cuda_kernels.cu already calls cudaDeviceSynchronize()
        Tensor C_gpu = A_gpu * B_gpu;
        
        benchmark::DoNotOptimize(C_gpu);
        
        // 3. Reset compute arena to avoid OOM
        globalArena.reset(); 
    }
}

BENCHMARK(BM_TensorAdd)->RangeMultiplier(2)->Range(256, 2048);
BENCHMARK(BM_NaiveMatMul)->RangeMultiplier(2)->Range(64, 512);

// DON'T FORGET TO REGISTER THE NEW BENCHMARK
BENCHMARK(BM_TiledMatMul)->RangeMultiplier(2)->Range(64, 512);

BENCHMARK_MAIN();