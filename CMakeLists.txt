cmake_minimum_required(VERSION 3.18)

# THIS MUST COME BEFORE project() !!!
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Now CMake will initialize nvcc with the correct GPU architectures
project(TensorGrad LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. FIND OPENMP
find_package(OpenMP REQUIRED)

# Aggressive compiler optimizations for the C++ side
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# Tell CMake where to look when a file says #include "Tensor.hpp"
include_directories(include)

# 2. THE CORE ENGINE
add_library(engine STATIC
    src/Arena.cpp
    src/Tensor.cpp
    src/Linear.cpp
    src/Loss.cpp
    src/Optimizer.cpp
    src/graph.cpp
    src/cuda_kernels.cu 
)

# 3. SURGICAL LINKING
# This forces the engine to compile and link with OpenMP, even if nvcc is the linker.
target_link_libraries(engine PUBLIC OpenMP::OpenMP_CXX)

# 4. EXECUTABLES
add_executable(train_xor examples/train_xor.cpp)
target_link_libraries(train_xor engine)
set_target_properties(train_xor PROPERTIES LINKER_LANGUAGE CUDA)

add_custom_target(run_xor
    COMMAND $<TARGET_FILE:train_xor>
    COMMAND dot -Tsvg forward_pass.dot -o graph.svg
    COMMAND xdg-open graph.svg || open graph.svg || start graph.svg || true
    DEPENDS train_xor
)

add_executable(train_circle examples/train_circle.cpp)
target_link_libraries(train_circle engine)
set_target_properties(train_circle PROPERTIES LINKER_LANGUAGE CUDA)

add_custom_target(run_circle
    COMMAND $<TARGET_FILE:train_circle>
    COMMAND dot -Tsvg forward_pass.dot -o graph.svg
    COMMAND xdg-open graph.svg || open graph.svg || start graph.svg || true
    DEPENDS train_circle
)

# --- BENCHMARKING SUITE ---
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing" FORCE) 

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main 
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(tensor_bench bench/tensor_bench.cpp)
target_link_libraries(tensor_bench PRIVATE engine benchmark::benchmark_main)
set_target_properties(tensor_bench PROPERTIES LINKER_LANGUAGE CUDA)