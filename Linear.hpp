#pragma once
#include "Tensor.hpp"
#include <vector>

class Linear {
    public:
        Tensor weights;
        Tensor bias;

        Tensor matmulCache;

        Linear(int inFeatures, int outFeatures);

        Tensor operator()(const Tensor& input);
        std::vector<Tensor*> parameters();
};