#include "graph.hpp"
#include "Tensor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <functional>
#include <cstdint>
#include <iomanip>

std::string shape_to_string(const std::vector<int>& shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    ss << "]";
    return ss.str();
}

std::string array_to_string(const double* arr, size_t size, size_t max_items = 3) {
    if (!arr || size == 0) return "[]";
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < std::min(size, max_items); i++) {
        if (arr[i] >= 0) ss << " "; 
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
    out << "  nodesep=0.6;\n";  
    out << "  ranksep=1.2;\n";  
    out << "  bgcolor=\"#1e1e2e\";\n"; 
    out << "  compound=true;\n"; 
    out << "  splines=spline;\n"; 
    
    out << "  node [fontname=\"Consolas,Monaco,monospace\", fontsize=11, shape=box, style=\"rounded,filled\", penwidth=0, margin=\"0.2,0.15\"];\n"; 
    out << "  edge [color=\"#a6adc860\", penwidth=2.0, arrowsize=0.7];\n";

    std::vector<const Tensor*> topo;
    std::set<const Tensor*> visited;
    std::map<const Tensor*, int> depths; 
    
    // NEW: We need to know who a node's children are so we can pull parameters closer to them
    std::map<const Tensor*, std::vector<const Tensor*>> children;

    // PASS 1: Calculate initial depths
    std::function<void(const Tensor*)> build_topo = [&](const Tensor* v) {
        if (!v) return;
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            int max_prev_depth = -1;
            for (const Tensor* parent : v->prev) {
                if (parent) {
                    children[parent].push_back(v); // Record the child-parent relationship
                    build_topo(parent);
                    if (depths[parent] > max_prev_depth) {
                        max_prev_depth = depths[parent];
                    }
                }
            }
            depths[v] = max_prev_depth + 1;
            topo.push_back(v);
        }
    };
    
    build_topo(root);

    // PASS 2: Fix the isolated parameters! 
    // If a node has no parents (like a Weight matrix), move its depth to exactly 1 stage before it gets used.
    for (const Tensor* node : topo) {
        if (node->prev.empty() && !children[node].empty()) {
            int min_child_depth = 999999;
            for (const Tensor* c : children[node]) {
                if (depths[c] < min_child_depth) {
                    min_child_depth = depths[c];
                }
            }
            depths[node] = min_child_depth - 1;
        }
    }

    std::map<int, std::vector<const Tensor*>> compute_stages;
    for (const Tensor* node : topo) {
        compute_stages[depths[node]].push_back(node);
    }

    for (const auto& [depth, nodes] : compute_stages) {
        out << "  subgraph cluster_" << depth << " {\n";
        out << "    label=\"STAGE " << depth << "\";\n";
        out << "    fontname=\"Consolas,Monaco,monospace\";\n";
        out << "    fontsize=10;\n";
        out << "    fontcolor=\"#6c7086\";\n"; 
        out << "    style=\"rounded,dashed\";\n";
        out << "    color=\"#45475a\";\n"; 
        out << "    margin=25;\n";

        for (const Tensor* node : nodes) {
            std::string uid = std::to_string(reinterpret_cast<uintptr_t>(node));
            
            std::string fillcolor = "#313244"; 
            std::string fontcolor = "#cdd6f4"; 
            std::string label_head = node->_op.empty() ? "Input / Param" : node->_op;
            
            if (node->_op.empty()) {
                fillcolor = "#f9e2af"; 
                fontcolor = "#11111b"; 
            } else if (node->_op == "ReLU") {
                fillcolor = "#a6e3a1"; 
                fontcolor = "#11111b";
            } else if (node->_op == "MSE" || node->_op == "sum") {
                fillcolor = "#f38ba8"; 
                fontcolor = "#11111b";
            } else {
                fillcolor = "#89b4fa"; 
                fontcolor = "#11111b";
            }

            std::string data_str = array_to_string(node->data, node->size);
            std::string grad_str = array_to_string(node->grad, node->size);

            out << "    \"" << uid << "\" [fillcolor=\"" << fillcolor << "\", fontcolor=\"" << fontcolor 
                << "\", label=\"" << label_head 
                << "\\nShape: " << shape_to_string(node->shape) 
                << "\\nData:  " << data_str
                << "\\nGrad:  " << grad_str
                << "\"];\n";
        }
        out << "  }\n"; 
    }

    for (const Tensor* node : topo) {
        std::string uid = std::to_string(reinterpret_cast<uintptr_t>(node));
        for (const Tensor* parent : node->prev) {
            if (!parent) continue;
            std::string p_uid = std::to_string(reinterpret_cast<uintptr_t>(parent));
            out << "  \"" << p_uid << "\" -> \"" << uid << "\";\n";
        }
    }

    out << "}\n";
    std::cout << "Graph exported to " << filename << ". Run: dot -Tsvg " << filename << " -o graph.svg\n";
}