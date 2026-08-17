#pragma once
#include <vector>
#include <functional>
#include <string>
#include "Device.hpp"

class Tensor {
public:
    size_t size;
    float* data;
    float* grad;
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<const Tensor*> prev;
    std::function<void(const float*)> _backward;
    std::string _op;

    Device device = Device::CPU;

    Tensor to(Device target_device) const;

    Tensor(const std::vector<int>& shape, bool isParam = false, Device device = Device::CPU);
    Tensor(const std::vector<float> data, const std::vector<int> shape, const std::vector<int> strides, bool isParam= false);
    // View constructor
    Tensor(float* data_ptr, float* grad_ptr, const std::vector<int>& shape, const std::vector<int>& strides, Device device); 

    float& at(const std::vector<int>& indices);
    Tensor broadcastTo(const std::vector<int>& targetShape) const;
    Tensor transpose() const;

    void zeroGrad();
    void backward();

    Tensor relu() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor pow(const float exp) const;
    Tensor sum() const;
};
