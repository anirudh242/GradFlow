#pragma once
#include <vector>
#include <functional>

class Tensor {
public:
    std::vector<double> data;
    mutable std::vector<double> grad;
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<const Tensor*> prev;
    std::function<void(const std::vector<double>&)> _backward;

    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<double> data, const std::vector<int> shape, const std::vector<int> strides);

    double& at(const std::vector<int>& indices);
    Tensor broadcastTo(const std::vector<int>& targetShape) const;

    void zeroGrad();

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
};
