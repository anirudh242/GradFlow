#include <benchmark/benchmark.h>
#include "Tensor.hpp"
#include "Arena.hpp" // CRITICAL: Need this to access globalArena
#include <vector>

static void BM_NaiveMatMul(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<double> dataA(size * size, 1.0);
    std::vector<double> dataB(size * size, 1.0);
    
    // 1. Setup: Allocate A and B in paramArena (isParam = true)
    // This protects them from the compute arena reset.
    Tensor A(dataA, {size, size}, {size, 1}, true);
    Tensor B(dataB, {size, size}, {size, 1}, true);

    // 2. Core Loop
    for (auto _ : state) {
        Tensor C = A * B; // Allocates C in globalArena
        
        benchmark::DoNotOptimize(C); 
        
        // 3. Reset the compute arena so we don't OOM!
        globalArena.reset(); 
    }
}

static void BM_TensorAdd(benchmark::State& state) {
    int size = state.range(0);
    
    std::vector<double> dataA(size * size, 1.0);
    std::vector<double> dataB(size * size, 2.0);
    
    // Allocate A and B in paramArena (isParam = true) to protect them from reset
    Tensor A(dataA, {size, size}, {size, 1}, true);
    Tensor B(dataB, {size, size}, {size, 1}, true);

    for (auto _ : state) {
        // This will call your new operator+
        Tensor C = A + B; 
        
        benchmark::DoNotOptimize(C); 
        
        // Reset the compute arena to prevent OOM errors on large sizes
        globalArena.reset(); 
    }
    paramArena.reset();
}

BENCHMARK(BM_TensorAdd)->RangeMultiplier(2)->Range(256, 2048);

BENCHMARK(BM_NaiveMatMul)->RangeMultiplier(2)->Range(64, 512);

BENCHMARK_MAIN();