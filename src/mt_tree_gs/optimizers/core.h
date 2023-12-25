#pragma once
#include "mt_tree_gs/core.h"
#include "helper_functions/core.h"

void backpropagateTreePrunning(MonteCarloTreeGeneticSearch *mctgs, size_t node_idx)
{
    // Function body goes here

    MTNode *node = &mctgs->nodes[node_idx];

    while (true)
    {

        if (node->parent == -1)
            break;
        node = &mctgs->nodes[node->parent];
    }

}; // Add the missing semicolon here
