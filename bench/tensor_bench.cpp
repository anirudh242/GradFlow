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

BENCHMARK(BM_NaiveMatMul)->RangeMultiplier(2)->Range(64, 512);

BENCHMARK_MAIN();