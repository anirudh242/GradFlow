#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include "Arena.hpp"

int main() {
    std::cout << "--- INITIALIZING TENSORS IN ARENA ---\n";
    
    // 1. Prediction: [3.0, 4.0]
    std::vector<int> shape = {2};
    std::vector<int> strides = {1};
    Tensor Pred(std::vector<double>{3.0, 4.0}, shape, strides);

    // 2. Target: [10.0, 10.0]
    Tensor Target(std::vector<double>{10.0, 10.0}, shape, strides);

    // --- FORWARD PASS ---
    Tensor Error = Pred - Target;               // [-7.0, -6.0]
    Tensor SqError = Error.pow(2.0);            // [49.0, 36.0]
    Tensor SumError = SqError.sum();            // [85.0]
    
    // Scale by 1/N to get the Mean
    double N = 2.0;
    Tensor Loss({1});
    Loss.data[0] = SumError.data[0] / N;        // 85.0 / 2 = 42.5
    Loss._op = "MSE";
    
    // Wire up the Loss to the graph
    Loss.prev.push_back(&SumError);
    Tensor* sum_ptr = &SumError;
    Loss._backward = [N, sum_ptr](const double* outGrad) {
        sum_ptr->grad[0] += (1.0 / N) * outGrad[0];
    };

    std::cout << "Loss: " << Loss.data[0] << " (Expected: 42.5)\n\n";

    // --- BACKWARD PASS ---
    std::cout << "--- BACKWARD PASS ---\n";
    Loss.grad[0] = 1.0; // Seed the root gradient!
    Loss.backward();

    std::cout << "Gradient of Pred[0]: " << Pred.grad[0] << " (Expected: -7)\n";
    std::cout << "Gradient of Pred[1]: " << Pred.grad[1] << " (Expected: -6)\n\n";

    // --- MEMORY SPEED TEST ---
    std::cout << "--- ARENA MEMORY CHECK ---\n";
    globalArena.print_usage(); // Should show some memory used by the intermediate nodes
    
    std::cout << "Triggering instant Epoch reset...\n";
    globalArena.reset();
    
    globalArena.print_usage(); // Should show exactly 0 doubles used

    return 0;
}