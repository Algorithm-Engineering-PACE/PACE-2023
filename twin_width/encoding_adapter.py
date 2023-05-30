from pathlib import Path
from typing import List

import pysat.solvers as slv
from networkx import Graph, number_connected_components, is_connected, complement

import twin_width.encoding as encoding
import twin_width.heuristic as heuristic

def run_graph_collection(graph_collection, original_g: Graph) -> List[tuple]:
    if graph_collection.cograph_contration_tree:
        return graph_collection.cograph_contration_tree

    enc = encoding.TwinWidthEncoding()
    contraction_sequence = []
    graph_collection.sort_by_tree_level()
    for prime in graph_collection.prime_set:
        if len(prime.graph.nodes) == number_connected_components(prime.graph):
            if len(prime.graph.nodes) == 2:
                contraction_sequence.append(tuple(prime.graph.nodes))
                continue
            prime.graph = complement(prime.graph)
        ub, mg = heuristic.get_ub(prime.graph)
        if not ub:
            for child, parent in mg.items():
                contraction_sequence.append((parent, child))
            continue
        ub2 = heuristic.get_ub2(prime.graph)
        ub = min(ub, ub2)
        _, od, parents, _ = enc.run(prime.graph, slv.Cadical, ub)
        for child in od[:-1]:
            contraction_sequence.append((parents[child], child))
    g = contract_graph_from_contraction_sequence(contraction_sequence, original_g.copy())
    if len(g.nodes) > 1: # we need to contract isolated vertices
        ub, mg = heuristic.get_ub(g)
        if not ub:
            for child, parent in mg.items():
                contraction_sequence.append((parent, child))
        else:
            raise Exception("Something went wrong")
    return contraction_sequence


def contract_graph_from_contraction_sequence(contraction_sequence: List[tuple], g) -> Graph:
    for parent, child in contraction_sequence:
        g.remove_node(child)
    return g
