#include "cuda_kernels.cuh"
#include <cuda_runtime.h>

#define TILE_SIZE 16 

__global__ void addKernel(const double* A, const double* B, double* C, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) { // prevent out of bounds
        C[i] = A[i] + B[i];
    }
}

__global__ void matMulKernel(const double* A, const double* B, double* C, size_t M, size_t K, size_t N) { // m rows, n cols, k inner dim
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; 
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    } 
}

__global__ void tiledMatmulKernel(const double* A, const double* B, double* C, size_t M, size_t K, size_t N) {
    __shared__ double sA[TILE_SIZE][TILE_SIZE];
    __shared__ double sB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x; // col 
    int ty = threadIdx.y; // row 

    size_t row = blockIdx.y * TILE_SIZE + ty;
    size_t col = blockIdx.x * TILE_SIZE + tx;

    double sum = 0.0;

    for (size_t t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // we use t as a multiplier to slide through chunks of 16 (TILE_SIZE) through the matrix
        if (row < M && t * TILE_SIZE + tx < K) {
            sA[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
        } else {
            sA[ty][tx] = 0.0;
        }
        if (t * TILE_SIZE + ty < K && col < N) {
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0;
        }
        __syncthreads();

        for (size_t i = 0; i < TILE_SIZE; i++) {
            sum += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" {
    void launchAddKernel(const double* A, const double* B, double* C, size_t N) {
        size_t sz = N * sizeof(double);

        // allocate vram
        double *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, sz);
        cudaMalloc((void**)&d_B, sz);
        cudaMalloc((void**)&d_C, sz);

        cudaMemcpy(d_A, A, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, sz, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256; // block size
        // if we did just N / threadsperblock, we could possibly get decimal values
        // eg: 10 / 4 = 2.5 which rounds off to 2.
        // this means that we would launch 4 * 2 = 8 threads leaving 2 unprocessed elements
        // to combat this you force it to round up by adding threadsPerBlock - 1.
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // grid size

        addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize(); // make cpu wait for kernel to finish before moving to next line
        
        cudaMemcpy(C, d_C, sz, cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void launchMatmulKernel(const double* A, const double* B, double* C, size_t M, size_t K, size_t N) {
        size_t sza = M * K * sizeof(double);
        size_t szb = N * K * sizeof(double);
        size_t szc = N * M * sizeof(double);
        double *d_A, *d_B, *d_C;

        cudaMalloc((void**)&d_A, sza);
        cudaMalloc((void**)&d_B, szb);
        cudaMalloc((void**)&d_C, szc);

        cudaMemcpy(d_A, A, sza, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, szb, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + 15) / 16, (M+15) / 16);

        matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();

        cudaMemcpy(C, d_C, szc, cudaMemcpyDeviceToHost); 

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    void launchTiledMatmulKernel(const double* A, const double* B, double* C, size_t M, size_t K, size_t N) {
        size_t sza = M * K * sizeof(double);
        size_t szb = N * K * sizeof(double);
        size_t szc = N * M * sizeof(double);
        double *d_A, *d_B, *d_C;

        cudaMalloc((void**)&d_A, sza);
        cudaMalloc((void**)&d_B, szb);
        cudaMalloc((void**)&d_C, szc);

        cudaMemcpy(d_A, A, sza, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, szb, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + 15) / 16, (M+15) / 16);

        tiledMatmulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();

        cudaMemcpy(C, d_C, szc, cudaMemcpyDeviceToHost); 

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    double* allocateVram(size_t bytes) {
        double* ptr = nullptr;
        cudaMalloc((void**)&ptr, bytes);
        return ptr;
    }

    void freeVram(double* ptr) {
        cudaFree(ptr);
    }

    void copyMemory(double* dst, const double* src, size_t bytes, bool toGpu) {
        if (toGpu) {
            cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        }
    }

    void fillZerosVram(double* ptr, size_t bytes) {
        cudaMemset(ptr, 0, bytes);
    }
}
