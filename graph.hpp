#pragma once
#include "Tensor.hpp"
#include <string>

std::string shape_to_string(const std::vector<int>& shape);
// Generates a Graphviz .dot file from a Tensor computation graph
void draw_graph(const Tensor* root, const std::string& filename = "graph.dot");