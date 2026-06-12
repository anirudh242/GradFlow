#include "Loss.hpp"

MSELoss::MSELoss() : error({1}), sqError({1}), sumError({1}) {}

Tensor MSELoss::operator()(const Tensor& pred, const Tensor& target) {
    // MSE = (1/N)*sum(yi'-yi)^2
    error = pred - target;
    sqError = error.pow(2.0);
    sumError = sqError.sum();
    
    double N = (double)pred.size; 
    
    Tensor outLoss({1});
    outLoss.data[0] = sumError.data[0] / N;
    outLoss._op = "MSE";
    
    outLoss.prev.push_back(&sumError);
    
    Tensor* sumPtr = &sumError;
    outLoss._backward = [N, sumPtr](const double* outGrad) {
        sumPtr->grad[0] += (1.0 / N) * outGrad[0];
    };
    
    return outLoss;
}
