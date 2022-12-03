import networkx as nx
import bz2


def parse(path):
    f = open(path)

    g = nx.Graph()

    p_line = False
    line = ""
    while not p_line:
        line = f.readline()
        if line and line[0] == 'p':
            p_line = True
    entries = line.strip().split()

    if len(entries) == 4:
        p, tww, num_of_vertices, num_of_edges = entries
        num_of_vertices = int(num_of_vertices)
        num_of_edges = int(num_of_edges)
    else:
        raise AttributeError("p line is not according to PACE format")

    g.add_nodes_from(range(1, num_of_vertices))
    for edge_index in range(num_of_edges):
        entries = f.readline().strip().split()
        g.add_edge(*[int(entire) for entire in entries])
    f.close()

    return g
