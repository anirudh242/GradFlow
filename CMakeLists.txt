cmake_minimum_required(VERSION 3.10)

project(autodiff_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. The Engine Block Optimizations
# -O3: Maximum compiler optimization (loop unrolling, vectorization). Mandatory for AI.
# -march=native: Compiles the math instructions specifically for your exact CPU architecture.
# -Wall -Wextra: Strict warning guards to catch dangling pointers or size_t mismatches.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

# 2. The Executable
# (Note: Only keep Optimizer.cpp here if you actually created a .cpp file for it. 
# If you only wrote Optimizer.hpp, remove it from this list).
add_executable(main
    main.cpp
    graph.cpp
    Tensor.cpp
)

# 3. The Execution Pipeline
# Uses $<TARGET_FILE:main> to guarantee CMake finds the compiled binary.
# Switched to .svg to preserve the crisp, dark-mode hacker aesthetic of the graph.
add_custom_target(run
    COMMAND $<TARGET_FILE:main>
    COMMAND dot -Tsvg forward_pass.dot -o graph.svg
    COMMAND xdg-open graph.svg
    DEPENDS main
)