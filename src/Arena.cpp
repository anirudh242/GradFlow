#include "Arena.hpp"

Arena globalArena(10000000);
Arena paramArena(10000000);

Arena::Arena(size_t maxElements = 10000000) : capacity(maxElements), offset(0), memory(std::make_unique<double[]>(maxElements)){}

double* Arena::allocate(size_t numElements) {
    if (offset + numElements > capacity)
        throw std::runtime_error("Arena out of memory, please increase capacity.");
    double* ptr = memory.get() + offset;
    offset += numElements;
    return ptr;
}

void Arena::reset() {
    offset = 0;
}

void Arena::print_usage() const {
    std::cout << "Arena usage: "  << offset << " / " << capacity << " doubles (" << (offset * 8.0 / 1024 / 1024) << " MB)\n";
}
