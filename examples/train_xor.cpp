#include <iostream>
#include <vector>
#include <iomanip>
#include "Tensor.hpp"
#include "Arena.hpp"
#include "Linear.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"
#include "graph.hpp"

int main() {
    std::cout << "STARTING XOR TRAINING\n";

    // DATASET
    std::vector<int> xShape = {4, 2};
    std::vector<int> xStrides = {2, 1};
    Tensor X(std::vector<double>{0,0, 0,1, 1,0, 1,1}, xShape, xStrides, true);

    std::vector<int> yShape = {4, 1};
    std::vector<int> yStrides = {1, 1};
    Tensor Y(std::vector<double>{0, 1, 1, 0}, yShape, yStrides, true);

    // MODEL ARCHITECTURE 
    Linear layer1(2, 16);
    Linear layer2(16, 1);
    MSELoss criterion;

    std::vector<Tensor*> params;
    auto l1Params = layer1.parameters();
    auto l2Params = layer2.parameters();
    params.insert(params.end(), l1Params.begin(), l1Params.end());
    params.insert(params.end(), l2Params.begin(), l2Params.end());

    SGD optimizer(params, 0.5); 

    Tensor test_h1 = layer1(X);
    Tensor test_a1 = test_h1.relu();
    Tensor test_pred = layer2(test_a1);
    Tensor test_loss = criterion(test_pred, Y);
    
    test_loss.grad[0] = 1.0;
    test_loss.backward();
    draw_graph(&test_loss, "forward_pass.dot");
    globalArena.reset(); 

    // TRAINING LOOP
    for (int epoch = 1; epoch <= 2000; epoch++) {
        Tensor h1 = layer1(X);
        Tensor a1 = h1.relu();
        Tensor pred = layer2(a1);
        
        Tensor loss = criterion(pred, Y);

        optimizer.zeroGrad();
        loss.grad[0] = 1.0;
        loss.backward();

        optimizer.step();
        globalArena.reset();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: ";
            if (loss.data[0] < 0.0001) {
                // Switch to scientific for microscopic numbers
                std::cout << std::scientific << std::setprecision(4) << loss.data[0] << "\n";
            } else {
                // Keep it clean for standard numbers
                std::cout << std::fixed << std::setprecision(4) << loss.data[0] << "\n";
            }
        }
    }

    std::cout << "\nFINAL PREDICTIONS\n";
    std::cout << std::fixed << std::setprecision(4); 
    Tensor h1 = layer1(X);
    Tensor a1 = h1.relu();
    Tensor finalPred = layer2(a1);
    
    for (int i = 0; i < 4; i++) {
        std::cout << "Input: [" << X.data[i*2] << ", " << X.data[i*2+1] << "] -> " << "Pred: " << finalPred.data[i] << " (Target: " << Y.data[i] << ")\n";
    }

    return 0;
}