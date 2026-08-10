#include "Tensor.hpp"
#include "Arena.hpp"
#include <string>
#include <stdexcept>
#include <set>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

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

Tensor::Tensor(const std::vector<int>& shape, bool isParam) : shape(shape) {
    size_t dataSize = 1;
    for (int i : shape) {
        dataSize *= i;  
    }
    size = dataSize;

    if (isParam) {
        data = paramArena.allocate(size);
        grad = paramArena.allocate(size);
    } else {
        data = globalArena.allocate(size);
        grad = globalArena.allocate(size);
    }

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
    const std::vector<int> strides,
    bool isParam
) : shape(shape), strides(strides) {
    size = input_data.size();
    if (isParam) {
        data = paramArena.allocate(size);
        grad = paramArena.allocate(size);
    } else {
        data = globalArena.allocate(size);
        grad = globalArena.allocate(size);
    }
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

// builds topo graph and calls _backward() for all nodes
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
    
    size_t res_size = result.size;

    // check if broadcasting occured 
    bool isAbroad = (this->size != res_size);
    bool isBbroad = (other.size != res_size);
    
    std::vector<int> resultStrides = result.strides;
    std::vector<int> broadAStrides = broadA.strides;
    std::vector<int> broadBStrides= broadB.strides;

    if (!isAbroad && !isBbroad) {
        const double* ptrA = this->data;
        const double* ptrB = other.data;
        double* ptrRes = result.data;
        
        long long total_len = res_size;
        long long aligned_len = total_len - (total_len % 4);

        #pragma omp parallel for
        for (long long i = 0; i < aligned_len; i += 4) {
            __m256d vecA = _mm256_loadu_pd(&ptrA[i]);
            __m256d vecB = _mm256_loadu_pd(&ptrB[i]);
            
            __m256d vecRes = _mm256_add_pd(vecA, vecB);
            
            _mm256_storeu_pd(&ptrRes[i], vecRes);
        }

        // scalar tail
        for (long long i = aligned_len; i < total_len; i++) {
            ptrRes[i] = ptrA[i] + ptrB[i];
        }
    } else {
        for (size_t flati = 0; flati < res_size; flati++) {
            size_t flatA = 0;
            size_t flatB = 0;

            for (size_t j = 0; j < commonShape.size(); j++) {
                int axis = (flati / resultStrides[j]) % commonShape[j];
                flatA += axis * broadAStrides[j];
                flatB += axis * broadBStrides[j];
            }
            
            result.data[flati] = broadA.data[flatA] + broadB.data[flatB];
        }
    }

    result.prev.push_back(this);
    result.prev.push_back(&other);

    double* grad_a = this->grad;
    double* grad_b = other.grad;

    result._backward = [grad_a, grad_b, isAbroad, isBbroad, 
                        commonShape, resultStrides, broadAStrides, broadBStrides, res_size](const double* outGrad) {
        // if no broadcasting occured we can directly map the grads
        if (!isAbroad && !isBbroad) {
            long long totalLen = res_size;
            long long alignedLen = totalLen - (totalLen % 4);
            
            #pragma omp parallel for
            for (long long i = 0; i < alignedLen; i += 4) {
                __m256d vecOut = _mm256_loadu_pd(&outGrad[i]);
                
                // a += incoming gradient
                __m256d vecGradA = _mm256_loadu_pd(&grad_a[i]);
                _mm256_storeu_pd(&grad_a[i], _mm256_add_pd(vecGradA, vecOut));
                
                // b += incoming gradient
                __m256d vecGradB = _mm256_loadu_pd(&grad_b[i]);
                _mm256_storeu_pd(&grad_b[i], _mm256_add_pd(vecGradB, vecOut));
            }
            
            // scalar tail 
            for (long long i = alignedLen; i < totalLen; i++) {
                grad_a[i] += outGrad[i];
                grad_b[i] += outGrad[i];
            }
            return; 
        }
        
        // cannot use openmp if broadcasted due to thread race conditions
        // multiple grads will go back to same idx which will corrupt the maths
        for (size_t flati = 0; flati < res_size; flati++) {
            size_t flatA = 0;
            size_t flatB = 0;

            for (size_t j = 0; j < commonShape.size(); j++) {
                int axis = (flati / resultStrides[j]) % commonShape[j];
                flatA += axis * broadAStrides[j];
                flatB += axis * broadBStrides[j];
            }
            
            grad_a[flatA] += outGrad[flati];
            grad_b[flatB] += outGrad[flati];
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

    int strideA_row = Ab.strides[Ab.strides.size() - 2];
    int strideA_col = Ab.strides[Ab.strides.size() - 1];
    
    int strideB_row = Bb.strides[Bb.strides.size() - 2];
    int strideB_col = Bb.strides[Bb.strides.size() - 1];

    int strideRes_row = result.strides[result.strides.size() - 2];
    int strideRes_col = result.strides[result.strides.size() - 1];

    const double* ptrA = Ab.data;
    const double* ptrB = Bb.data;
    double* ptrRes = result.data;

    constexpr int BLOCK_SIZE = 32;

    for (int b = 0; b < totalBatches; b++) {
        size_t batchOffsetA = 0;
        size_t batchOffsetB = 0;
        size_t batchOffsetRes = 0;

        for (size_t i = 0; i < finalBatchShape.size(); i++) {
            int coord = (b / batchStrides[i]) % finalBatchShape[i];
            batchOffsetA += coord * Ab.strides[i];
            batchOffsetB += coord * Bb.strides[i];
            batchOffsetRes += coord * result.strides[i];
        }

        for (int r = 0; r < rowsA; r++) {
            for (int c = 0; c < colsB; c++) {
                size_t idxRes = batchOffsetRes + r * strideRes_row + c * strideRes_col;
                ptrRes[idxRes] = 0.0;
            }
        }

        // outer loops move 64x64 tiles; br, bk, bc = boundaries of curr tile
        // using openmp to use all 16 cores (multithreading)
        #pragma omp parallel for
        for (int br = 0; br < rowsA; br += BLOCK_SIZE) {
            for (int bk = 0; bk < colsA; bk += BLOCK_SIZE) {
                for (int bc = 0; bc < colsB; bc += BLOCK_SIZE) {
                    // prevent index out of bounds
                    int r_end = std::min(br + BLOCK_SIZE, rowsA);
                    int k_end = std::min(bk + BLOCK_SIZE, colsA);
                    int c_end = std::min(bc + BLOCK_SIZE, colsB);

                    // matmul loops
                    for (int r = br; r < r_end; r++) {
                        for (int k = bk; k < k_end; k++) {
                            size_t idxA = batchOffsetA + r * strideA_row + k * strideA_col;
                            double a_val = ptrA[idxA]; 
                            
                            int c = bc; 
                            
                            // avx2 path
                            // if mems not flat then (reading a col over a row) then avx will grab the wrong data
                            if (strideB_col == 1 && strideRes_col == 1) {
                                // Broadcast a_val to [a, a, a, a]
                                __m256d vec_a = _mm256_set1_pd(a_val); // copy a_val 4 times into 256 bit register
                                
                                for (; c <= c_end - 4; c += 4) {
                                    size_t idxB = batchOffsetB + k * strideB_row + c;
                                    size_t idxRes = batchOffsetRes + r * strideRes_row + c;
                                    
                                    __m256d vec_b = _mm256_loadu_pd(&ptrB[idxB]); // load from mem into 256 bit register
                                    __m256d vec_res = _mm256_loadu_pd(&ptrRes[idxRes]);
                                    
                                    __m256d vec_mul = _mm256_mul_pd(vec_a, vec_b); // multiplication with simd
                                    vec_res = _mm256_add_pd(vec_res, vec_mul);
                                    
                                    _mm256_storeu_pd(&ptrRes[idxRes], vec_res); // back to ram
                                }
                            }
                            
                            // scalar tail
                            // avx2 works in batches of 4 so if cols % 4 != 0 then there will be leftover cols.
                            // leftover cols are processed individually with normal multiplication 
                            for (; c < c_end; c++) {
                                size_t idxB = batchOffsetB + k * strideB_row + c * strideB_col;
                                size_t idxRes = batchOffsetRes + r * strideRes_row + c * strideRes_col;
                                
                                ptrRes[idxRes] += a_val * ptrB[idxB];
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<int> resShape = result.shape;
    std::vector<int> resStrides = result.strides;
    size_t res_size = result.size;

    result.prev.push_back(this);
    result.prev.push_back(&other);

    const Tensor* self = this;
    const Tensor* other_ptr = &other;

    result._backward = [self, other_ptr, resShape, resStrides, res_size](const double* outGrad) {
        Tensor dC(resShape); 
        
        #pragma omp parallel for
        for(size_t i = 0; i < res_size; i++) {
            dC.data[i] = outGrad[i];
        }

        Tensor At = self->transpose();
        Tensor Bt = other_ptr->transpose();
        Tensor dA = dC * Bt;
        Tensor dB = At * dC;

        #pragma omp parallel for
        for (size_t i = 0; i < self->size; i++) {
            self->grad[i] += dA.data[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < other_ptr->size; i++) {
            other_ptr->grad[i] += dB.data[i];
        }
    };

    result._op = "*";

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    // broadcasting
    std::vector<int> commonShape = broadcastShapes(shape, other.shape);
    Tensor broadA = broadcastTo(commonShape);
    Tensor broadB = other.broadcastTo(commonShape);

    Tensor result(commonShape);
    
    size_t res_size = result.size;

    // check if broadcasting occured 
    bool isAbroad = (this->size != res_size);
    bool isBbroad = (other.size != res_size);
    
    std::vector<int> resultStrides = result.strides;
    std::vector<int> broadAStrides = broadA.strides;
    std::vector<int> broadBStrides= broadB.strides;

    if (!isAbroad && !isBbroad) {
        const double* ptrA = this->data;
        const double* ptrB = other.data;
        double* ptrRes = result.data;
        
        long long total_len = res_size;
        long long aligned_len = total_len - (total_len % 4);

        #pragma omp parallel for
        for (long long i = 0; i < aligned_len; i += 4) {
            __m256d vecA = _mm256_loadu_pd(&ptrA[i]);
            __m256d vecB = _mm256_loadu_pd(&ptrB[i]);
            
            __m256d vecRes = _mm256_sub_pd(vecA, vecB);
            
            _mm256_storeu_pd(&ptrRes[i], vecRes);
        }

        // scalar tail
        for (long long i = aligned_len; i < total_len; i++) {
            ptrRes[i] = ptrA[i] + ptrB[i];
        }
    } else {
        for (size_t flati = 0; flati < res_size; flati++) {
            size_t flatA = 0;
            size_t flatB = 0;

            for (size_t j = 0; j < commonShape.size(); j++) {
                int axis = (flati / resultStrides[j]) % commonShape[j];
                flatA += axis * broadAStrides[j];
                flatB += axis * broadBStrides[j];
            }
            
            result.data[flati] = broadA.data[flatA] - broadB.data[flatB];
        }
    }

    result.prev.push_back(this);
    result.prev.push_back(&other);

    double* grad_a = this->grad;
    double* grad_b = other.grad;

    result._backward = [grad_a, grad_b, isAbroad, isBbroad, 
                        commonShape, resultStrides, broadAStrides, broadBStrides, res_size](const double* outGrad) {
        
        // if no broadcasting occured we can directly map the grads
        if (!isAbroad && !isBbroad) {
            long long totalLen = res_size;
            long long alignedLen = totalLen - (totalLen % 4);
            
            #pragma omp parallel for
            for (long long i = 0; i < alignedLen; i += 4) {
                __m256d vecOut = _mm256_loadu_pd(&outGrad[i]);
                
                // a += incoming gradient
                __m256d vecGradA = _mm256_loadu_pd(&grad_a[i]);
                _mm256_storeu_pd(&grad_a[i], _mm256_add_pd(vecGradA, vecOut));
                
                // b -= incoming gradient
                __m256d vecGradB = _mm256_loadu_pd(&grad_b[i]);
                _mm256_storeu_pd(&grad_b[i], _mm256_sub_pd(vecGradB, vecOut));
            }
            
            // scalar tail 
            for (long long i = alignedLen; i < totalLen; i++) {
                grad_a[i] += outGrad[i];
                grad_b[i] -= outGrad[i];
            }
            return; 
        }
        
        // cannot use openmp if broadcasted due to thread race conditions
        // multiple grads will go back to same idx which will corrupt the maths
        for (size_t flati = 0; flati < res_size; flati++) {
            size_t flatA = 0;
            size_t flatB = 0;

            for (size_t j = 0; j < commonShape.size(); j++) {
                int axis = (flati / resultStrides[j]) % commonShape[j];
                flatA += axis * broadAStrides[j];
                flatB += axis * broadBStrides[j];
            }
            
            grad_a[flatA] += outGrad[flati];
            grad_b[flatB] -= outGrad[flati];
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

    double* gradIn = this->grad;
    const double* dataIn = this->data;
    size_t sz = this->size;

    result._backward = [gradIn, dataIn, sz, exp](const double* outGrad) {
        for (size_t i = 0; i < sz; i++) {
            double derivative = exp * std::pow(dataIn[i], exp-1.0);
            gradIn[i] += outGrad[i] * derivative;
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

    double* gradIn = this->grad;
    size_t sz = this->size;
    result._backward = [gradIn, sz](const double* outGrad) {
        for (size_t i = 0; i < sz; i++)
            gradIn[i] += 1.0 * outGrad[0];
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

    double* gradIn = this->grad;
    const double* dataIn = this->data;
    size_t sz = this->size;

    result._backward = [gradIn, dataIn, sz](const double* outGrad) {
        for (size_t i = 0; i < sz; i++) {
            double localDer = (dataIn[i] > 0.0) ? 1.0 : 0.0;
            gradIn[i] += outGrad[i] * localDer;
        }
    };
    
    result._op = "ReLU";
    return result;
}