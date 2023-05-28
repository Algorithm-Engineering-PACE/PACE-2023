import sys

import networkx as nx
from networkx import Graph
from typing import List
import pysat.solvers as slv

import preprocessing
import twin_width.heuristic as heuristic
import twin_width.encoding as encoding

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


def process_graph(graph : Graph ,instance_name = None, save_result_to_csv = False) -> dict:
    ## our preprocessing
    g, contraction_tree, is_cograph = preprocessing.preproccess(graph)
    if is_cograph:
        return {"instance_name": instance_name
                        ,"num_of_nodes": graph.number_of_nodes()
                        ,"num_of_edges": graph.number_of_edges()
                        ,"tww": 0
                        ,"elimination_ordering": set()
                        ,"contraction_tree": contraction_tree
                        ,"cycle_times": None
                        ,"duration": None
                                }
    ub = heuristic.get_ub(g)
    ub2 = heuristic.get_ub2(g)
    ub = min(ub, ub2)

    enc = encoding.TwinWidthEncoding()
    cb, od, mg, times = enc.run(g, slv.Cadical, ub)
    contraction_tree.update(mg)

    return  {
        "instance_name": instance_name
        ,"num_of_nodes": g.number_of_nodes()
        ,"num_of_edges": g.number_of_edges()
        ,"tww": cb
        ,"elimination_ordering": od
        ,"contraction_tree": contraction_tree
        ,"cycle_times": times
    }


def print_contraction_tree_from_input():
    g = parse_stdin()
    result = process_graph(g.copy())
    print_contraction_tree(result["contraction_tree"],
        result["elimination_ordering"],
        result["tww"], g.number_of_nodes())

def print_contraction_tree(parents: dict, ordering: list, tww: int , num_of_nodes_orginal_graph, print_to_file = False, file_path = None):
    symetric_diff = set(i for i in range(1,num_of_nodes_orginal_graph + 1)).symmetric_difference(set(ordering))
    lines = []
    if parents:
        if len(parents) == int(num_of_nodes_orginal_graph) - 1:
            if tww > 0:
            ## child is contracted to parnet
                if symetric_diff:
                    for child in symetric_diff:
                        line = f"{parents[child]} {child}"
                        print(line, flush=True)
                        lines.append(str(line)+"\n")

                for child in ordering[:-1]:
                    line = f"{parents[child]} {child}"
                    print(line, flush=True)
                    lines.append(str(line)+"\n")
            else:
                for parent,child in dict(parents).items():
                    line = f"{child} {parent}"
                    print(line, flush=True)
                    lines.append(str(line)+"\n")

            if print_to_file and file_path:
                with open(file_path, 'w') as file:
                    file.writelines(lines)

        else:
            raise Exception("contraction tree is not valid - number of contraction != num_of_nodes - 1")
    else:
        raise Exception("result is not valid")

# @dataclass
# class PrimeGraphCollection:
#     prime_graphs: List[Graph]
#     cograph_contration_tree: List[tuple] # TODO: change decode function in
