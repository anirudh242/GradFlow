#include "Linear.hpp"
#include <random>
#include <cmath>

Linear::Linear(int inFeatures, int outFeatures) : 
    weights({inFeatures, outFeatures}, true), 
    bias({1, outFeatures}, true), 
    matmulCache({1})
{
    std::mt19937 gen(42);
    float limit = std::sqrt(6.0 / (inFeatures + outFeatures));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < weights.size; i++) {
        weights.data[i] = dist(gen);
    }
    for (size_t i = 0; i < bias.size; i++) {
        bias.data[i] = 0.0;
    }
}

Tensor Linear::operator()(const Tensor& input) {
    matmulCache = input * weights; 
    return matmulCache + bias;
}

std::vector<Tensor*> Linear::parameters() {
    return { &weights, &bias };
}
