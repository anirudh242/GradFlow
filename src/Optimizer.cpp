#include "Optimizer.hpp"

SGD::SGD(std::vector<Tensor*> params, float learningRate) : parameters(params), lr(learningRate) {}

void SGD::zeroGrad() {
    for (Tensor* p : parameters) {
        p->zeroGrad();
    }
}

void SGD::step() {
    for (Tensor* p : parameters) {
        for (size_t i = 0; i < p->size; i++) {
            p->data[i] -= lr * p->grad[i];
        }
    }
}