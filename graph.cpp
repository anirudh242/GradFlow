#include "graph.hpp"
#include "Tensor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <functional>
#include <cstdint>
#include <iomanip> // Required for decimal precision

// Formats the shape array into a string "[32, 64]"
std::string shape_to_string(const std::vector<int>& shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    ss << "]";
    return ss.str();
}

// Safely grabs the first few elements of a raw double* array so huge matrices don't break the UI
std::string array_to_string(const double* arr, size_t size, size_t max_items = 4) {
    if (!arr || size == 0) return "[]";
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < std::min(size, max_items); i++) {
        // Round to 4 decimal places for clean UI
        ss << std::fixed << std::setprecision(4) << arr[i];
        if (i < std::min(size, max_items) - 1) ss << ", ";
    }
    if (size > max_items) ss << ", ...";
    ss << "]";
    return ss.str();
}

void draw_graph(const Tensor* root, const std::string& filename) {
    std::ofstream out(filename);
    out << "digraph G {\n";
    out << "  rankdir=LR;\n"; 
    out << "  bgcolor=\"#1e1e2e\";\n"; 
    out << "  node [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10, shape=record, style=\"rounded,filled\", color=\"#45475a\"];\n"; 
    out << "  edge [color=\"#a6adc8\", penwidth=1.5];\n";

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
        std::string uid = std::to_string(reinterpret_cast<uintptr_t>(node));
        
        std::string fillcolor = "#313244"; 
        std::string fontcolor = "#cdd6f4"; 
        std::string label_head = node->_op;
        
        if (node->_op.empty()) {
            label_head = "Input/Param";
            fillcolor = "#f9e2af"; // Yellow
            fontcolor = "#11111b"; 
        } else if (node->_op == "ReLU") {
            fillcolor = "#a6e3a1"; // Green
            fontcolor = "#11111b";
        } else if (node->_op == "MSE" || node->_op == "sum") {
            fillcolor = "#f38ba8"; // Red
            fontcolor = "#11111b";
        } else {
            fillcolor = "#89b4fa"; // Blue for Math (+, -, *)
            fontcolor = "#11111b";
        }

        // Generate the data and gradient strings
        std::string data_str = array_to_string(node->data, node->size);
        std::string grad_str = array_to_string(node->grad, node->size);

        // Record syntax uses '{ }' and '|' to stack text vertically into clean UI rows
        out << "  \"" << uid << "\" [fillcolor=\"" << fillcolor << "\", fontcolor=\"" << fontcolor 
            << "\", label=\"{ " 
            << label_head 
            << " | Shape: " << shape_to_string(node->shape) 
            << " | data: " << data_str
            << " | grad: " << grad_str
            << " }\"];\n";

        for (const Tensor* parent : node->prev) {
            if (!parent) continue;
            std::string p_uid = std::to_string(reinterpret_cast<uintptr_t>(parent));
            out << "  \"" << p_uid << "\" -> \"" << uid << "\";\n";
        }
    }

    out << "}\n";
}