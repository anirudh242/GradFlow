#pragma once
#include <vector>
#include <functional>
#include <string>

class Tensor {
public:
    std::vector<double> data;
    mutable std::vector<double> grad;
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<const Tensor*> prev;
    std::function<void(const std::vector<double>&)> _backward;
    std::string _op;

    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<double> data, const std::vector<int> shape, const std::vector<int> strides);

    double& at(const std::vector<int>& indices);
    Tensor broadcastTo(const std::vector<int>& targetShape) const;
    Tensor transpose() const;

    void zeroGrad();
    void backward();

    Tensor relu() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor pow(const double exp) const;
    Tensor sum() const;
};
