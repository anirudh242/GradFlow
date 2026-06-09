#pragma once
#include "Tensor.hpp"

class MSELoss {
    private:
        Tensor error;
        Tensor sqError;
        Tensor sumError;

    public:
        MSELoss();
        Tensor operator()(const Tensor& pred, const Tensor& target);
};