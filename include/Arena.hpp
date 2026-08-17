#pragma once
#include <memory>
#include "Device.hpp"

class Arena {
private:
  size_t capacity;

  size_t cpuOffset;
  std::unique_ptr<double[]> cpuMemory;

  size_t gpuOffset;
  double* gpuMemory;

public:
  Arena(size_t maxElements = 10000000);
  ~Arena();

  double *allocate(size_t numElements, Device device = Device::CPU);

  void reset();
  void print_usage() const;
};

extern Arena globalArena;
extern Arena paramArena;
