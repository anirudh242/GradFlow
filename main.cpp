#include <iostream>
#include "Tensor.hpp"

int main() {
    // A: Shape [2, 2, 3] (Two 2x3 matrices)
    Tensor A({2, 2, 3});
    // Batch 0
    A.at({0, 0, 0}) = 1; A.at({0, 0, 1}) = 2; A.at({0, 0, 2}) = 3;
    A.at({0, 1, 0}) = 4; A.at({0, 1, 1}) = 5; A.at({0, 1, 2}) = 6;
    // Batch 1
    A.at({1, 0, 0}) = 7; A.at({1, 0, 1}) = 8; A.at({1, 0, 2}) = 9;
    A.at({1, 1, 0}) = 1; A.at({1, 1, 1}) = 2; A.at({1, 1, 2}) = 3;

    // B: Shape [2, 3, 2] (Two 3x2 matrices)
    Tensor B({2, 3, 2});
    // Batch 0
    B.at({0, 0, 0}) = 1; B.at({0, 0, 1}) = 2;
    B.at({0, 1, 0}) = 3; B.at({0, 1, 1}) = 4;
    B.at({0, 2, 0}) = 5; B.at({0, 2, 1}) = 6;
    // Batch 1
    B.at({1, 0, 0}) = 2; B.at({1, 0, 1}) = 1;
    B.at({1, 1, 0}) = 4; B.at({1, 1, 1}) = 3;
    B.at({1, 2, 0}) = 6; B.at({1, 2, 1}) = 5;

    // The Magic N-Dimensional Execution
    Tensor C = A * B;

    std::cout << "Result Shape: {" << C.shape[0] << ", " << C.shape[1] << ", " << C.shape[2] << "}\n";
    std::cout << "Batch 0, Row 0, Col 0: " << C.at({0, 0, 0}) << "\n";
    std::cout << "Batch 1, Row 0, Col 0: " << C.at({1, 0, 0}) << "\n";

    return 0;
}