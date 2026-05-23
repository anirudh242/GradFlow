#include <iostream>
#include "Tensor.hpp"

int main() {
    // A is a 1D Vector [3]
    Tensor A({3});
    A.at({0}) = 1.0; A.at({1}) = 2.0; A.at({2}) = 3.0;

    // B is a 2D Matrix [2, 3]
    Tensor B({2, 3});
    B.at({0,0})=1; B.at({0,1})=1; B.at({0,2})=1;
    B.at({1,0})=1; B.at({1,1})=1; B.at({1,2})=1;

    // Forward Pass builds the graph
    Tensor C = A + B;

    // Seed the output gradient (pretend the network had an error of 1.0 everywhere)
    std::fill(C.grad.begin(), C.grad.end(), 1.0);

    // Fire the engine backwards!
    C._backward(C.grad);

    std::cout << "Gradient of A[0]: " << A.grad[0] << "\n";
    std::cout << "Gradient of A[1]: " << A.grad[1] << "\n";
    std::cout << "Gradient of A[2]: " << A.grad[2] << "\n";

    return 0;
}