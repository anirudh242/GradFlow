#pragma once
#include "Tensor.hpp"
#include <string>

std::string shape_to_string(const std::vector<int>& shape);
void draw_graph(const Tensor* root, const std::string& filename = "graph.dot");