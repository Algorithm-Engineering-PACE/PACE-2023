import sys

import networkx as nx, Graph
from typing import List
import pysat.solvers as slv

from preprocessing import preproccess, NodeType
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


def solve_md_tree(md_tree, g):
    if(md_tree.node_type ==  NodeType.NORMAL):
        return ([], md_tree.children[0])
    output_list = []
    for child in md_tree.children:
            output_list.append(solve_md_tree(child,g))
    subg = []
    cur_contr_seq = []
    for (contr_seq, root_node) in output_list:
            cur_contr_seq += contr_seq
            subg.append(root_node)
    if (md_tree.node_type == NodeType.PRIME):
        cur_g = g.subgraph(subg)
        assert len(cur_g.nodes)>3
        cb, od, ct, times = run_solver(cur_g)
        contraction_tree = create_sequence_from_dict(ct,od)
        cur_contr_seq.extend(contraction_tree)
        return (cur_contr_seq, root_node) # root_id
    if (md_tree.node_type == NodeType.PARALLEL or md_tree.node_type == NodeType.SERIES):
        for i in range(1,len(subg)):
            cur_contr_seq.append((subg[0], subg[i]))
        return (cur_contr_seq, subg[0])

def run_solver(g):
    ub = heuristic.get_ub(g)
    ub2 = heuristic.get_ub2(g)
    ub = min(ub, ub2)

    enc = encoding.TwinWidthEncoding()
    cb, od, mg, times = enc.run(g, slv.Cadical103, ub)
    return cb, od, mg, times


def create_sequence_from_dict(parents: dict, ordering: list):
    contraction_tree = []
    for child in ordering[:-1]:
        contraction_tree.append((parents[child],child))
    return contraction_tree

def print_contraction_tree(contraction_tree,g_num_of_nodes, print_to_file = False, file_path = None):
    lines = []
    if len(contraction_tree) == g_num_of_nodes - 1:
        for parent,child in contraction_tree:
            line = f"{parent} {child}"
            print(line, flush=True)
            lines.append(str(line)+"\n")
        if print_to_file and file_path:
            with open(file_path, 'w') as file:
                file.writelines(lines)
    else:
           raise Exception("c - contraction tree is not valid - number of contraction != num_of_nodes - 1")



def process_graph(graph : Graph ,instance_name = None, save_result_to_csv = False) -> dict:
    ## our preprocessing
    cb,od,times = None,None,None
    md_tree, is_prime_graph = preproccess(graph)
    if is_prime_graph:
        cb, od, cs, times = run_solver(graph)
        contraction_tree = create_sequence_from_dict(cs,od)

    else:
        contraction_tree,root = solve_md_tree(md_tree,graph)
    return  {
        "instance_name": instance_name
        ,"num_of_nodes": graph.number_of_nodes()
        ,"num_of_edges": graph.number_of_edges()
        ,"tww": cb
        ,"elimination_ordering": od
        ,"contraction_tree": contraction_tree
        ,"cycle_times": times
    }


def print_contraction_tree_from_input():
    g = parse_stdin()
    result = process_graph(g.copy())
    contraction_tree = result["contraction_tree"]
    print_contraction_tree(contraction_tree,g.number_of_nodes())