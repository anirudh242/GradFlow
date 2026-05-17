#include "Tensor.hpp"
#include <string>
#include <stdexcept>

// Helper to find broadcasted shape of 2 shapes
std::vector<int> broadcastShapes(const std::vector<int>& shapeA, const std::vector<int>& shapeB) {
    int ndimA = shapeA.size();
    int ndimB = shapeB.size();
    int outndim = std::max(ndimA, ndimB);
    
    std::vector<int> outShape(outndim);
    
    for (int i = 0; i < outndim; i++) {
        int dimA = (ndimA - 1 - i >= 0) ? shapeA[ndimA - 1 - i] : 1;
        int dimB = (ndimB - 1 - i >= 0) ? shapeB[ndimB - 1 - i] : 1;
        
        if (dimA == dimB) {
            outShape[outndim - 1 - i] = dimA;
        } else if (dimA == 1) {
            outShape[outndim - 1 - i] = dimB;
        } else if (dimB == 1) {
            outShape[outndim - 1 - i] = dimA;
        } else {
            throw std::runtime_error("Shapes are not broadcastable.");
        }
    }
    return outShape;
}

Tensor::Tensor(const std::vector<int>& s) : shape(s) {
    int dataSize = 1;
    for (int i : shape) {
        dataSize *= i;  
    }
    data.assign(dataSize, 0.0);

    strides.resize(shape.size());
    int currStride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = currStride;
        currStride *= shape[i];
    }
}

Tensor::Tensor(
    const std::vector<double> data, 
    const std::vector<int> shape, 
    const std::vector<int> strides
) : data(data), shape(shape), strides(strides) {}

double& Tensor::at(const std::vector<int>& indices) {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("No. of indices does not match rank");
    }
    
    int flatIndex = 0;
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds for dimension: " + std::to_string(i));
        }

        flatIndex += indices[i] * strides[i];
    }

    return data[flatIndex];
}

Tensor Tensor::broadcastTo(const std::vector<int>& targetShape) const {
    int ndimCurrent = shape.size();
    int ndimTarget = targetShape.size();
    if (ndimTarget < ndimCurrent)
        throw std::runtime_error("Target shape can't have less dimensions than current shape");

    std::vector<int> newStrides(targetShape.size(), 0);

    for (int i = 0; i < ndimTarget; i++) {
        // at i = 0 these point to the back of the arrays.
        // we are traversing them backwards
        int targetidx = ndimTarget - i - 1;
        int currentidx = ndimCurrent - i - 1;

        // if our currentidx is -1 (i.e there is no matching dim to targetidx) then we imagine it as 1. [3] = [1, 3] (1 row 3 cols)
        int targetDimSize = targetShape[targetidx];
        int currDimSize = (currentidx >= 0) ? shape[currentidx] : 1; 
       
        if (currDimSize == targetDimSize)
            newStrides[targetidx] = (currentidx >= 0) ? strides[currentidx] : 0;
        else if (currDimSize == 1)
            newStrides[targetidx] = 0;
        else
            throw std::runtime_error("Shapes can't be broadcasted");
    }

    return Tensor(data, targetShape, newStrides);
}

Tensor Tensor::operator+(const Tensor& other) const {
    // broadcasting
    std::vector<int> commonShape = broadcastShapes(this->shape, other.shape);
    Tensor broadA = this->broadcastTo(commonShape);
    Tensor broadB = other.broadcastTo(commonShape);

    Tensor result(commonShape);

    int total = 1;
    for (int dim : commonShape)
        total *= dim;
    
    std::vector<int> curr(commonShape.size(), 0);
    
    for (int flati = 0; flati < total; flati++) {
        for (int j = 0; j < commonShape.size(); j++) {
            curr[j] = (flati / result.strides[j]) % commonShape[j];    
        }

        result.at(curr) = broadA.at(curr) + broadB.at(curr);
    }

    return result;
}
