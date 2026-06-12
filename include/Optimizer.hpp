#pragma once
#include "Tensor.hpp"
#include <vector>

class SGD {
public:
    std::vector<Tensor*> parameters;
    double lr;

    SGD(std::vector<Tensor*> params, double learningRate);

    void zeroGrad();
    void step();
};