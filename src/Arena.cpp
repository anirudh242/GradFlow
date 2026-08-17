#include "Arena.hpp"
#include "cuda_kernels.cuh"
#include <iostream>
#include <stdexcept>

Arena globalArena(100000000);
Arena paramArena(100000000);

Arena::Arena(size_t maxElements)
    : capacity(maxElements), 
      cpuOffset(0),
      cpuMemory(std::make_unique<double[]>(maxElements)),
      gpuOffset(0),
      gpuMemory(nullptr) {}

Arena::~Arena() {
    if (gpuMemory != nullptr) {
        freeVram(gpuMemory);
    }
}

double *Arena::allocate(size_t numElements, Device device) {
    if (device == Device::CPU) {
        if (cpuOffset + numElements > capacity)
            throw std::runtime_error(
                "Arena out of memory, please increase capacity.");
        double *ptr = cpuMemory.get() + cpuOffset;
        cpuOffset += numElements;
        return ptr;
    } else {
        if (gpuOffset + numElements > capacity)
            throw std::runtime_error("Arena GPU out of memory, please increase capacity.");
        
        if (gpuMemory == nullptr) {
            std::cout << "Arena initializing " << (capacity * 8.0 / 1024 / 1024) << "MB out of VRAM" << std::endl;
            gpuMemory = allocateVram(capacity * sizeof(double));
            if (gpuMemory == nullptr) {
                throw std::runtime_error("Fatal: cudaMalloc failed to allocate VRAM.");
            }
        }

        double *ptr = gpuMemory;
        gpuOffset += numElements;
        return ptr;
    }
}

void Arena::reset() {
    cpuOffset = 0; 
    gpuOffset = 0;
}

void Arena::print_usage() const {
    std::cout << "Arena CPU usage: " << cpuOffset << " / " << capacity << " ("
              << (cpuOffset * 8.0 / 1024 / 1024) << " MB)\n";
    if (gpuMemory != nullptr) {
        std::cout << "Arena GPU usage: " << gpuOffset << " / " << capacity << " ("
                  << (gpuOffset * 8.0 / 1024 / 1024) << " MB)\n";
    }
}
