#pragma once
#include <vector>

class Tensor {
public:
    std::vector<double> data;
    std::vector<int> shape;
    std::vector<int> strides;

    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<double> data, const std::vector<int> shape, const std::vector<int> strides);

    double& at(const std::vector<int>& indices);
    Tensor broadcastTo(const std::vector<int>& targetShape) const;

    Tensor operator+(const Tensor& other) const;
};
