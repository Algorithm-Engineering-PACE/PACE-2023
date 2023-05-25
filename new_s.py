import sys

import networkx as nx
from networkx import Graph
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


def process_graph(graph : Graph ,instance_name = None,save_result_to_csv = False):
    ## our preprocessing
    res = preprocessing.preproccess(graph)
    g = res['output_graph']
    if res.get("cograph"):
        return ({"instance_name": instance_name
                        ,"num_of_nodes": graph.number_of_nodes()
                        ,"num_of_edges": graph.number_of_edges()
                        ,"tww": 0
                        ,"elimination_ordering": None
                        ,"contraction_tree": res['cograph']['contraction_tree']
                        ,"cycle_times": None
                        ,"duration": None
                                })
    ub = heuristic.get_ub(g)
    ub2 = heuristic.get_ub2(g)
    ub = min(ub, ub2)

    enc = encoding.TwinWidthEncoding()
    cb = enc.run(g, slv.Cadical103, ub)

    return({"instance_name": instance_name
                    ,"num_of_nodes": g.number_of_nodes()
                    ,"num_of_edges": g.number_of_edges()
                    ,"tww": cb[0]
                    ,"elimination_ordering": cb[1]
                    ,"contraction_tree": cb[2]
                    ,"cycle_times": cb[3]
                            })


def proccess_graph_from_input():
    g = parse_stdin()
    result = process_graph(g.copy())
    contraction_tree = dict(result).get("contraction_tree")
    num_of_nodes = dict(result).get("num_of_nodes")
    if contraction_tree and num_of_nodes:
        if len(contraction_tree) == int(num_of_nodes) - 1:
            ## y is contracted to x
            for y,x in dict(contraction_tree).items():
                print(f"{x} {y}", flush=True)
        else:
            raise Exception("contraction tree is not valid - number of contraction != num_of_nodes - 1")
    else:
        raise Exception("result is not valid")

if __name__=='__main__':
    proccess_graph_from_input()

