#pragma once
#include "Tensor.hpp"
#include <string>

// Generates a Graphviz .dot file from a Tensor computation graph
void draw_graph(const Tensor* root, const std::string& filename = "graph.dot");