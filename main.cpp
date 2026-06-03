#include <iostream>
#include <vector>
#include <numeric>
#include "Tensor.hpp"

int main() {
    std::cout << "--- FORWARD PASS ---\n";

    // 1. Inputs (A): Shape [2, 3], initialized to 1.0
    Tensor A({2, 3});
    std::fill(A.data.begin(), A.data.end(), 1.0);

    // 2. Weights (B): Shape [3, 2], initialized to 2.0
    Tensor B({3, 2});
    std::fill(B.data.begin(), B.data.end(), 2.0);

    // 3. Bias (C): Shape [2], initialized to [1.0, 2.0]
    Tensor C({2});
    C.data[0] = 1.0; 
    C.data[1] = 2.0;

    // 4. The Math: Y = (A * B) + C
    // We explicitly capture the Matmul intermediate so we can trigger its lambda
    Tensor M = A * B;
    Tensor Y = M + C;

    std::cout << "Result Y Shape: [" << Y.shape[0] << ", " << Y.shape[1] << "]\n";
    std::cout << "Y[0, 0]: " << Y.at({0, 0}) << " (Expected: 7)\n";
    std::cout << "Y[0, 1]: " << Y.at({0, 1}) << " (Expected: 8)\n\n";

    std::cout << "--- BACKWARD PASS (MANUAL TRANSMISSION) ---\n";

    // 1. Seed the gradient of the final output with 1.0 (The "Error")
    std::fill(Y.grad.begin(), Y.grad.end(), 1.0);

    // 2. Shift Gear 1: Backpropagate through Addition
    // This pushes gradients into M.grad and C.grad
    if (Y._backward) {
        Y._backward(Y.grad);
    }

    // 3. Shift Gear 2: Backpropagate through Matmul
    // This pushes gradients into A.grad and B.grad
    if (M._backward) {
        M._backward(M.grad);
    }

    // --- PROOF OF CALCULUS ---
    std::cout << "Gradient of Bias C: [" << C.grad[0] << ", " << C.grad[1] << "]\n";
    std::cout << "  -> Expected: [2, 2] (Because batch size is 2, it accumulated twice!)\n\n";

    std::cout << "Gradient of Weights B[0, 0]: " << B.grad[0] << "\n";
    std::cout << "  -> Expected: 2 (A^T * dM)\n\n";

    std::cout << "Gradient of Input A[0, 0]: " << A.grad[0] << "\n";
    std::cout << "  -> Expected: 4 (dM * B^T)\n";

    return 0;
}