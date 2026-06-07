#include <iostream>
#include <vector>
#include "Tensor.hpp"

int main() {
    // Input: One positive, one negative
    Tensor Input({1, 2});
    Input.data = {5.0, -3.0};

    // Forward Pass
    Tensor Activated = Input.relu();

    std::cout << "--- FORWARD PASS ---\n";
    std::cout << "Activated[0]: " << Activated.data[0] << " (Expected: 5)\n";
    std::cout << "Activated[1]: " << Activated.data[1] << " (Expected: 0)\n\n";

    // Backward Pass
    Activated.grad[0] = 10.0; // Simulate an incoming gradient of 10.0
    Activated.grad[1] = 10.0; 
    Activated.backward();

    std::cout << "--- BACKWARD PASS ---\n";
    std::cout << "Input Grad[0]: " << Input.grad[0] << " (Expected: 10)\n";
    std::cout << "Input Grad[1]: " << Input.grad[1] << " (Expected: 0)\n";

    return 0;
}