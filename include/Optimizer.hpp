#pragma once
#include "Tensor.hpp"
#include <vector>

class SGD {
public:
    std::vector<Tensor*> parameters;
    float lr;

    SGD(std::vector<Tensor*> params, float learningRate);

    void zeroGrad();
    void step();
};