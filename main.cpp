#include <iostream>
#include <vector>
#include <random>
#include "Tensor.hpp"
#include "Arena.hpp"
#include "Linear.hpp"
#include "Loss.hpp"
#include "Optimizer.hpp"

int main() {
    std::cout << "--- GENERATING CIRCLE DATASET ---\n";
    
    int numSamples = 200;
    std::vector<double> xData(numSamples * 2);
    std::vector<double> yData(numSamples * 1);

    std::mt19937 gen(123); 
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for(int i = 0; i < numSamples; i++) {
        double x1 = dist(gen);
        double x2 = dist(gen);
        xData[i*2] = x1;
        xData[i*2+1] = x2;
        yData[i] = ((x1 * x1) + (x2 * x2) < 0.36) ? 1.0 : 0.0;
    }

    std::vector<int> xShape = {numSamples, 2};
    std::vector<int> xStrides = {2, 1};
    Tensor X(xData, xShape, xStrides, true); 

    std::vector<int> yShape = {numSamples, 1};
    std::vector<int> yStrides = {1, 1};
    Tensor Y(yData, yShape, yStrides, true); 

    Linear layer1(2, 16);
    Linear layer2(16, 16);
    Linear layer3(16, 1);
    MSELoss criterion;

    std::vector<Tensor*> params;
    auto l1p = layer1.parameters();
    auto l2p = layer2.parameters();
    auto l3p = layer3.parameters();
    params.insert(params.end(), l1p.begin(), l1p.end());
    params.insert(params.end(), l2p.begin(), l2p.end());
    params.insert(params.end(), l3p.begin(), l3p.end());

    SGD optimizer(params, 0.05);

    Tensor test_h1 = layer1(X);
    Tensor test_a1 = test_h1.relu();
    Tensor test_h2 = layer2(test_a1);
    Tensor test_a2 = test_h2.relu();
    Tensor test_pred = layer3(test_a2);
    Tensor test_loss = criterion(test_pred, Y);
    
    
    globalArena.reset(); 

    std::cout << "--- STARTING DEEP TRAINING ---\n";

    for (int epoch = 1; epoch <= 4000; epoch++) {
        Tensor h1 = layer1(X);
        Tensor a1 = h1.relu();
        Tensor h2 = layer2(a1);
        Tensor a2 = h2.relu();
        Tensor pred = layer3(a2);
        
        Tensor loss = criterion(pred, Y);

        optimizer.zeroGrad();
        loss.grad[0] = 1.0;
        loss.backward();

        optimizer.step();
        globalArena.reset();

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss.data[0] << "\n";
        }
    }

    return 0;
}