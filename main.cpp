#include <iostream>
#include "Tensor.hpp"

int main() {
    Tensor weights({2, 3});
    weights.at({0, 0}) = 1.0; weights.at({0, 1}) = 2.0; weights.at({0, 2}) = 3.0;
    weights.at({1, 0}) = 4.0; weights.at({1, 1}) = 5.0; weights.at({1, 2}) = 6.0;

    Tensor bias({3});
    bias.at({0}) = 10.0; bias.at({1}) = 20.0; bias.at({2}) = 30.0;

    Tensor output = weights + bias;

    std::cout << "Output Shape: {" << output.shape[0] << ", " << output.shape[1] << "}\n";
    std::cout << "Row 0: " << output.at({0, 0}) << ", " << output.at({0, 1}) << ", " << output.at({0, 2}) << "\n";
    std::cout << "Row 1: " << output.at({1, 0}) << ", " << output.at({1, 1}) << ", " << output.at({1, 2}) << "\n";

    return 0;
}