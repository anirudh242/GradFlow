#include "Tensor.hpp"
#include "Arena.hpp"
#include <string>
#include <stdexcept>
#include <set>
#include <cmath>

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
    size_t dataSize = 1;
    for (int i : shape) {
        dataSize *= i;  
    }
    size = dataSize;
    data = globalArena.allocate(size);
    grad = globalArena.allocate(size);

    for (size_t i = 0; i < size; i++) {
        data[i] = 0.0;
        grad[i] = 0.0;
    }

    strides.resize(shape.size());
    int currStride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
        strides[i] = currStride;
        currStride *= shape[i];
    }
}

Tensor::Tensor(
    const std::vector<double> input_data, 
    const std::vector<int> shape, 
    const std::vector<int> strides
) : shape(shape), strides(strides) {
    size = input_data.size();
    data = globalArena.allocate(size);
    grad = globalArena.allocate(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = input_data[i];
        grad[i] = 0.0;
    }
}

double& Tensor::at(const std::vector<int>& indices) {
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("No. of indices does not match rank");
    }
    
    size_t flatIndex = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds for dimension: " + std::to_string(i));
        }

        flatIndex += (size_t)indices[i] * (size_t)strides[i];
    }

    return data[flatIndex];
}

Tensor Tensor::transpose() const {
    int ndim = shape.size();
    if (ndim < 2) 
        return *this;

    std::vector<int> newShape = shape;
    std::vector<int> newStrides = strides;

    // batches stay the same, rows and cols of 2d matrices swap
    std::swap(newShape[ndim - 1], newShape[ndim - 2]);
    std::swap(newStrides[ndim - 1], newStrides[ndim - 2]);

    std::vector<double> vec_data(data, data + size);
    return Tensor(vec_data, newShape, newStrides);
}

void Tensor::zeroGrad() {
    for (size_t i = 0; i < size; i++) {
        grad[i] = 0.0;
    }
}

void Tensor::backward() {
    std::vector<const Tensor*> topo;
    std::set<const Tensor*> visited;

    std::function<void(const Tensor*)> build_topo = [&](const Tensor* v) {
        if (visited.find(v) == visited.end()){
            visited.insert(v);
            for (const Tensor* child : v->prev) {
                if (!child) continue;
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(this);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const Tensor* current_node = *it;
        if (current_node->_backward) {
            current_node->_backward(current_node->grad);
        }
    }
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

    std::vector<double> vec_data(data, data + size);
    return Tensor(vec_data, targetShape, newStrides);
}

Tensor Tensor::operator+(const Tensor& other) const {
    // broadcasting
    std::vector<int> commonShape = broadcastShapes(shape, other.shape);
    Tensor broadA = broadcastTo(commonShape);
    Tensor broadB = other.broadcastTo(commonShape);

    Tensor result(commonShape);

    size_t total = 1;
    for (int dim : commonShape)
        total *= dim;
    
    std::vector<int> curr(commonShape.size(), 0);
    
    for (size_t flati = 0; flati < total; flati++) {
        for (size_t j = 0; j < commonShape.size(); j++) {
            curr[j] = (flati / result.strides[j]) % commonShape[j];    
        }

        result.at(curr) = broadA.at(curr) + broadB.at(curr);
    }

    result.prev.push_back(this);
    result.prev.push_back(&other);

    std::vector<int> resultStrides = result.strides;
    std::vector<int> broadAStrides = broadA.strides;
    std::vector<int> broadBStrides= broadB.strides;
    size_t res_size = result.size;

    result._backward = [this, &other, commonShape, resultStrides, broadAStrides, broadBStrides, res_size](const double* outGrad) {
        for (size_t flati = 0; flati < res_size; flati++) {
            size_t flatA = 0;
            size_t flatB = 0;

            for (size_t j = 0; j < commonShape.size(); j++) {
                int axis = (flati / resultStrides[j]) % commonShape[j];
                flatA += axis * broadAStrides[j];
                flatB += axis * broadBStrides[j];
            }
            
            grad[flatA] += 1.0 * outGrad[flati];
            other.grad[flatB] += 1.0 * outGrad[flati];
        }
    };

    result._op = "+";

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    int colsA = shape[shape.size() - 1];
    int rowsA = shape[shape.size() - 2];
    int colsB = other.shape[other.shape.size() - 1];
    int rowsB = other.shape[other.shape.size() - 2];

    if (colsA != rowsB)
        throw std::runtime_error("Inner dimensions do not match");
        
    std::vector<int> batchShapeA(
        shape.begin(), 
        shape.end() >= shape.begin() + 2 ? shape.end() - 2 : shape.begin()
    );
    std::vector<int> batchShapeB(
        other.shape.begin(), 
        other.shape.end() >= other.shape.begin() + 2 ? other.shape.end() - 2: other.shape.begin()
    );
    std::vector<int> finalBatchShape = broadcastShapes(batchShapeA, batchShapeB);

    std::vector<int> targetShapeA = finalBatchShape;
    targetShapeA.push_back(rowsA);
    targetShapeA.push_back(colsA);
    Tensor Ab = broadcastTo(targetShapeA);

    std::vector<int> targetShapeB = finalBatchShape;
    targetShapeB.push_back(rowsB);
    targetShapeB.push_back(colsB);
    Tensor Bb = other.broadcastTo(targetShapeB);

    std::vector<int> finalShape = finalBatchShape;
    finalShape.push_back(rowsA);
    finalShape.push_back(colsB);
    Tensor result(finalShape);

    int totalBatches = 1;
    for (int dim : finalBatchShape) {
        totalBatches *= dim;
    }

    std::vector<int> batchStrides(finalBatchShape.size(), 0);
    int currBatchStride = 1;
    for (int i = finalBatchShape.size() - 1; i >= 0; i--) {
        batchStrides[i] = currBatchStride;
        currBatchStride *= finalBatchShape[i];
    }

    std::vector<int> batchcoords(finalBatchShape.size(), 0);
    for (int b = 0; b < totalBatches; b++) {
        
        for (size_t i = 0; i < finalBatchShape.size(); i++) {
            batchcoords[i] = (b / batchStrides[i]) % finalBatchShape[i];
        }
        for (int r = 0; r < rowsA; r++) {
            for (int c = 0; c < colsB; c++) {
                double sum = 0.0;
                for (int k = 0; k < colsA; k++) {
                    std::vector<int> coordA = batchcoords;
                    coordA.push_back(r);
                    coordA.push_back(k);

                    std::vector<int> coordB = batchcoords;
                    coordB.push_back(k);
                    coordB.push_back(c);

                    sum += Ab.at(coordA) * Bb.at(coordB);
                }

                std::vector<int> coordresult = batchcoords;
                coordresult.push_back(r);
                coordresult.push_back(c);
                result.at(coordresult) = sum;
            }
        }
    }

    std::vector<int> resShape = result.shape;
    std::vector<int> resStrides = result.strides;
    size_t res_size = result.size;

    result.prev.push_back(this);
    result.prev.push_back(&other);

    result._backward = [this, &other, resShape, resStrides, res_size](const double* outGrad) {
        std::vector<double> outGradVec(outGrad, outGrad + res_size);
        Tensor dC(outGradVec, resShape, resStrides);
        Tensor At = transpose();
        Tensor Bt = other.transpose();
        Tensor dA = dC * Bt;
        Tensor dB = At * dC;
        for (size_t i = 0; i < size; i++) {
            grad[i] += dA.data[i];
        }
        for (size_t i = 0; i < other.size; i++)
        {
            other.grad[i] += dB.data[i];
        }
        
    };

    result._op = "*";

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    std::vector<int> commonShape = broadcastShapes(shape, other.shape);
    Tensor broadA = broadcastTo(commonShape);
    Tensor broadB = other.broadcastTo(commonShape);

    Tensor result(commonShape);

    size_t total = 1;
    for (int dim : commonShape)
        total *= dim;
    
    std::vector<int> curr(commonShape.size(), 0);
    
    for (size_t flati = 0; flati < total; flati++) {
        for (size_t j = 0; j < commonShape.size(); j++) {
            curr[j] = (flati / result.strides[j]) % commonShape[j];    
        }

        result.at(curr) = broadA.at(curr) - broadB.at(curr);
    }

    result.prev.push_back(this);
    result.prev.push_back(&other);

    std::vector<int> resultStrides = result.strides;
    std::vector<int> broadAStrides = broadA.strides;
    std::vector<int> broadBStrides= broadB.strides;
    size_t res_size = result.size;

    result._backward = [this, &other, commonShape, resultStrides, broadAStrides, broadBStrides, res_size](const double* outGrad) {
        for (size_t flati = 0; flati < res_size; flati++) {
            size_t flatA = 0;
            size_t flatB = 0;

            for (size_t j = 0; j < commonShape.size(); j++) {
                int axis = (flati / resultStrides[j]) % commonShape[j];
                flatA += axis * broadAStrides[j];
                flatB += axis * broadBStrides[j];
            }
            
            grad[flatA] += 1.0 * outGrad[flati];
            other.grad[flatB] -= 1.0 * outGrad[flati];
        }
    };

    result._op = "-";

    return result;
}

Tensor Tensor::pow(const double exp) const {
    Tensor result(shape);
    
    for (size_t i = 0; i < size; i++) {
        result.data[i] = std::pow(data[i], exp);
    }
    result.prev.push_back(this);

    result._backward = [this, exp](const double* outGrad) {
        for (size_t i = 0; i < size; i++) {
            double derivative = exp * std::pow(data[i], exp-1.0);
            grad[i] += outGrad[i] * derivative;
        }
    };

    result._op = "^" + std::to_string(exp);
    return result;
}

Tensor Tensor::sum() const {
    Tensor result({1});
    double total = 0.0;
    for (size_t i = 0; i < size; i++)
        total += data[i];
    result.data[0] = total;

    result.prev.push_back(this);

    result._backward = [this](const double* outGrad) {
        for (size_t i = 0; i < size; i++)
            grad[i] += 1.0 * outGrad[0];
    };

    result._op = "sum";
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape);

    for (size_t i = 0; i < size; i++) {
        result.data[i] = (data[i] > 0.0) ? data[i] : 0.0;
    }

    result.prev.push_back(this);

    result._backward = [this](const double* outGrad) {
        for (size_t i = 0; i < size; i++) {
            double localDer = (data[i] > 0.0) ? 1.0 : 0.0;
            grad[i] += outGrad[i] * localDer;
        }
    };
    
    result._op = "ReLU";
    return result;
}