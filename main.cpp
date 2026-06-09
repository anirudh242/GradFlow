#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include "Arena.hpp"

int main() {
    std::cout << "--- INITIALIZING TWO-ARENA TEST ---\n";
    
    std::vector<int> shape = {2};
    std::vector<int> strides = {1};

    // 1. PERSISTENT PARAMETERS (is_param = true)
    // These live in paramArena and should survive the epoch wipe.
    Tensor Weights(std::vector<double>{0.5, -0.5}, shape, strides, true);
    Tensor Bias(std::vector<double>{0.1, 0.1}, shape, strides, true);

    // 2. EPHEMERAL DATA (is_param = false, which is default)
    // These live in globalArena and wipe every epoch.
    Tensor Input(std::vector<double>{2.0, 4.0}, shape, strides);
    Tensor Target(std::vector<double>{1.0, 0.0}, shape, strides);

    std::cout << "\n--- MEMORY BEFORE MATH ---\n";
    std::cout << "Param Arena: "; paramArena.print_usage();
    std::cout << "Global Arena: "; globalArena.print_usage();

    // 3. FORWARD PASS
    // All intermediate math nodes automatically allocate in globalArena
    Tensor Pred = Input + Weights; // Simulated layer
    Tensor Error = Pred - Target;
    Tensor SqError = Error.pow(2.0);
    Tensor Loss = SqError.sum();

    // 4. BACKWARD PASS
    Loss.grad[0] = 1.0; // Seed the root
    Loss.backward();

    std::cout << "\n--- MEMORY AFTER BACKWARD PASS ---\n";
    std::cout << "Param Arena (Should only hold W and B): "; paramArena.print_usage();
    std::cout << "Global Arena (Should hold all intermediate math): "; globalArena.print_usage();

    // 5. THE EPOCH WIPE
    std::cout << "\n--- TRIGGERING EPOCH RESET ---\n";
    globalArena.reset();

    std::cout << "Param Arena (Must NOT be 0): "; paramArena.print_usage();
    std::cout << "Global Arena (Must be exactly 0): "; globalArena.print_usage();

    return 0;
}