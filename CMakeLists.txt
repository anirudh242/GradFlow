cmake_minimum_required(VERSION 3.10)

# Name your framework
project(autodiff_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Aggressive compiler optimizations (-O3 and -march=native) for maximum speed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -fopenmp")

# 1. THE PUBLIC API
# Tell CMake where to look when a file says #include "Tensor.hpp"
include_directories(include)

# 2. THE CORE ENGINE
# Compile the private implementation into a highly optimized Static Library
add_library(engine STATIC
    src/Arena.cpp
    src/Tensor.cpp
    src/Linear.cpp
    src/Loss.cpp
    src/Optimizer.cpp
    src/graph.cpp
)

# 3. EXAMPLE 1: XOR
add_executable(train_xor examples/train_xor.cpp)
target_link_libraries(train_xor engine)

# Custom target to run XOR and pop open the graph
add_custom_target(run_xor
    COMMAND $<TARGET_FILE:train_xor>
    COMMAND dot -Tsvg forward_pass.dot -o graph.svg
    COMMAND xdg-open graph.svg || open graph.svg || start graph.svg || true
    DEPENDS train_xor
)

# 4. EXAMPLE 2: CIRCLE DATASET
add_executable(train_circle examples/train_circle.cpp)
target_link_libraries(train_circle engine)

# Custom target to run the Circle dataset and pop open the graph
add_custom_target(run_circle
    COMMAND $<TARGET_FILE:train_circle>
    COMMAND dot -Tsvg forward_pass.dot -o graph.svg
    COMMAND xdg-open graph.svg || open graph.svg || start graph.svg || true
    DEPENDS train_circle
)

# --- BENCHMARKING SUITE ---
include(FetchContent)

# Prevent Google Benchmark from pulling in GoogleTest and bloating your build
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing" FORCE) 

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main 
)
FetchContent_MakeAvailable(googlebenchmark)

# Compile the benchmark executable
add_executable(tensor_bench bench/tensor_bench.cpp)
# Link against your 'engine' static library and benchmark_main
target_link_libraries(tensor_bench PRIVATE engine benchmark::benchmark_main)
