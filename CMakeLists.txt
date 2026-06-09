cmake_minimum_required(VERSION 3.10)

project(autodiff_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

add_executable(main
    main.cpp
    graph.cpp
    Tensor.cpp
    Arena.cpp
    Optimizer.cpp
    Loss.cpp
    Linear.cpp
)

# Custom target to compile, run, build the graph, and open it
add_custom_target(run
    COMMAND $<TARGET_FILE:main>
    COMMAND dot -Tsvg forward_pass.dot -o graph.svg
    # Cross-platform open command that fails silently if the OS doesn't support one of them
    COMMAND xdg-open graph.svg || open graph.svg || start graph.svg || true
    DEPENDS main
)