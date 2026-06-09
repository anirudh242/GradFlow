#pragma once
#include <memory>
#include <iostream>

class Arena {
    private:
        size_t capacity;
        size_t offset;
        std::unique_ptr<double[]> memory;

    public:
        Arena(size_t maxElements);

        double* allocate(size_t numElements);
        void reset();
        void print_usage() const; 

};

extern Arena globalArena;
extern Arena paramArena;
