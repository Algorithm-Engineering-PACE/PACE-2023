import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from networkx import Graph
from typing import List, Optional
import pysat.solvers as slv

import preprocessing
import twin_width.heuristic as heuristic
import twin_width.encoding as encoding
from twin_width.encoding_adapter import run_graph_collection
from pace_verifier import check_sequence, read_graph, read_sequence

def parse_stdin():
    g = nx.Graph()
    # Read input from stdin
    for index,line in enumerate(sys.stdin):
        try:
            line = line.decode('ascii')
        except AttributeError:
            pass
        entries = line.strip().split()
        if index == 0:
            num_of_nodes = int(entries[2])
            g.add_nodes_from(list(i for i in range(1,num_of_nodes + 1)))
        else:
            if len(entries) == 2:
                try:
                    x , y = int(entries[0]) ,int(entries[1])
                except ValueError:
                    continue
                g.add_edge(x, y)

    return g


def process_graph(graph : Graph, graph_path: Optional[Path] = None) -> dict:
    ## our preprocessing
    res = preprocessing.preproccess(graph)
    contraction_seq = run_graph_collection(res, graph)
    d = None
    if graph_path:
        pace_g = read_graph(graph_path)
        d = check_sequence(pace_g, contraction_seq)


    return  {
        "instance_path": graph_path
        ,"num_of_nodes": graph.number_of_nodes()
        ,"num_of_edges": graph.number_of_edges()
        ,"tww": d
        ,"contraction_tree": contraction_seq
    }


def print_contraction_tree_from_input():
    g = parse_stdin()
    result = process_graph(g.copy())
    print_contraction_tree(result["contraction_tree"], g.number_of_nodes())

def print_contraction_tree(contraction_sequance, num_of_nodes_orginal_graph):
    if len(contraction_sequance) == int(num_of_nodes_orginal_graph) - 1:
        for p,v in contraction_sequance:
            print(f"{p} {v}")
    else:
        raise Exception("contraction tree is not valid - number of contraction != num_of_nodes - 1")

@dataclass
class Prime:
    graph: Graph
    tree_level: int
@dataclass
class PrimeGraphCollection:
    prime_set: List[Prime]
    cograph_contration_tree: List[tuple]

    def sort_by_tree_level(self):
            self.prime_set.sort(key=lambda prime: prime.tree_level, reverse=True)
