#include "graph.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <functional>
#include <cstdint>

// A quick helper to format the shape array into a string "[32, 64]"
std::string shape_to_string(const std::vector<int>& shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    ss << "]";
    return ss.str();
}

void draw_graph(const Tensor* root, const std::string& filename) {
    std::ofstream out(filename);
    out << "digraph G {\n";
    out << "  rankdir=LR;\n"; // Left to Right layout
    out << "  node [shape=record, style=filled, fillcolor=\"#282a36\", fontcolor=\"#f8f8f2\", color=\"#6272a4\"];\n"; 
    out << "  edge [color=\"#6272a4\"];\n";

    std::vector<const Tensor*> topo;
    std::set<const Tensor*> visited;
    
    std::function<void(const Tensor*)> build_topo = [&](const Tensor* v) {
        if (!v) return;
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const Tensor* child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    
    build_topo(root);

    for (const Tensor* node : topo) {
        // Use the raw memory address as the unique ID for Graphviz
        std::string uid = std::to_string(reinterpret_cast<uintptr_t>(node));
        
        std::string op_label = node->_op.empty() ? "Input/Weight" : node->_op;

        out << "  \"" << uid << "\" [label=\"{ " 
            << "Shape: " << shape_to_string(node->shape) 
            << " | op: " << op_label
            << " }\"];\n";

        for (const Tensor* parent : node->prev) {
            if (!parent) continue;
            std::string p_uid = std::to_string(reinterpret_cast<uintptr_t>(parent));
            out << "  \"" << p_uid << "\" -> \"" << uid << "\";\n";
        }
    }

    out << "}\n";
    std::cout << "Graph exported to " << filename << ". Run: dot -Tsvg " << filename << " -o graph.svg\n";
}