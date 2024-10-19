import networkx as nx


if __name__ == "__main__":
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 2), (2, 4), (4, 1), (4, 5), (4, 6), (5, 7), (5, 8), (6, 8), (7, 9), (8, 10), (9, 7), (9, 10)])

    # Get predecessors for each node
    for node in range(1, 11):
        predecessors = [pred for pred in G.predecessors(node)]
        print(f"Predecessors of node {node}: {predecessors}")
