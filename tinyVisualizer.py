from tinyParser import tinyParser
from graphviz import Digraph
from tinyTokenizer import TokenKind
from collections import defaultdict
# import graphviz
# print(graphviz.__file__)

def visualize(parser: tinyParser):
    dot = Digraph()

    # # TokenKind Duplicate Check:
    # tokens = defaultdict(list)
    # for token in TokenKind.__iter__():
    #     tokens[token.value].append(token.name)
    #     if len(tokens[token.value]) >= 2:
    #         raise SyntaxError(f"{tokens[token.value]} are duplicates at number {token.value}")
    # print(tokens)

    # block settings
    for block in parser.tinyBlocks:
        block_name = "BB" + str(block.block_id)
        block_label = "<b>" + block_name + "|"
        block_label += "{"
        for i in range(len(block.SSA_ids)):
            ssa_id = block.SSA_ids[i]
            if i != 0:
                block_label += "|"
            block_label += str(parser.SSAs[ssa_id])
        block_label += "}"
        dot.node(name = block_name, label = block_label, shape='record')

    # relationship settings
    for block in parser.tinyBlocks:
        # basic edges
        block_name = "BB" + str(block.block_id)
        if block.fall_through_block:
            fall_through_block_name = "BB" + str(block.fall_through_block)
            dot.edge(block_name, fall_through_block_name, label='fall-through')
        if block.branching_block:
            branch_block_name = "BB" + str(block.branching_block)
            dot.edge(block_name, branch_block_name, label='b')
        if block.function_calls:
            for func_id in block.function_calls:
                func_block_name = "BB" + str(func_id)
                dot.edge(block_name, func_block_name, label='func call')

        # # domination edges
        # for dom in block.dominates:
        #     dom_name = "BB" + str(dom)
        #     dot.edge(block_name, dom_name, label = "dom", color='red', style='dotted')
        # if block.dominated_by is not None:
        #     dot.edge(block_name, "BB" + str(block.dominated_by), label = "by", color='blue', style='dotted')

    # Get the DOT representation as a string
    dot_representation = dot.source

    # Write the DOT representation to a file
    with open('graph_output.dot', 'w') as file:
        file.write(dot_representation)

    # # --------------------------------------------------------------#
    #
    # # dominator tree visualization
    # dot = Digraph()
    #
    # # block settings
    # for block in parser.tinyBlocks:
    #     block_name = "BB" + str(block.block_id)
    #     block_label = block_name
    #     dot.node(name = block_name, label = block_label)
    #
    # # relationship settings
    # for block in parser.tinyBlocks:
    #     # basic edges
    #     block_name = "BB" + str(block.block_id)
    #
    #     for dom in block.dominates:
    #         dom_name = "BB" + str(dom)
    #         dot.edge(block_name, dom_name, label = "dom")
    #
    #     if block.dominated_by is not None:
    #         dot.edge(block_name, "BB" + str(block.dominated_by), label = "by")
    #
    # # Get the DOT representation as a string
    # dot_representation = dot.source
    #
    # # Write the dominance tree to a file
    # with open('dominators.dot', 'w') as file:
    #     file.write(dot_representation)