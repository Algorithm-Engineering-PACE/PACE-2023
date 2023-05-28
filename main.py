import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pysat.solvers as slv
from pandas import DataFrame
import typer

import twin_width.encoding as encoding
import twin_width.encoding_signed_bipartite as encoding_signed_bipartite
import twin_width.heuristic as heuristic
import twin_width.parser as parser
import preprocessing
import tools
from utils import print_contraction_tree_from_input, process_graph, print_contraction_tree
from logger import logger


from networkx import graph


app = typer.Typer()
BASE_PATH = Path(__file__).parent

@app.command()
def clean_results():
    delete_files_starting_with("results_tww")

@app.command()
def process_graphs_from_dir(instance_path: Path, start: int =0, to: int=-1):

    instance_path = (BASE_PATH / instance_path).resolve()
    if not os.path.exists(instance_path):
        logger.error(f"folder is not exists {instance_path}")
    files = sorted(os.listdir(instance_path))
    filter(lambda file_name: file_name.endswith(".gr"), files)
    results = []
    for file_name in files:
        res = process_file(instance_path, file_name, save_result_to_csv=False)
        results.append(res)

    df = DataFrame().from_records(results)
    results_file_name = (BASE_PATH /
        f"results_tww_{datetime.now()}.csv").resolve()
    df.to_csv(results_file_name)

@app.command()
def process_graph_from_instance(file_name: Path):
    process_file(BASE_PATH, file_name)

@app.command()
def proccess_graph_from_stdin():
    g = parser.parse_stdin()
    print_contraction_tree_from_input(g.copy())


def process_file(instance_path: Path, file_name: str ,
    save_result_to_csv = True,save_pace_output = True):
    instance_file_name = (instance_path / file_name).resolve().as_posix()

    output_graphs = False
    if any(x == "-l" for x in sys.argv[1:-1]):
        output_graphs = True

    if instance_file_name.endswith(".cnf"):
        g = parser.parse_cnf(file_name)
        ub = heuristic.get_ub2_polarity(g)

        logger.debug(f"UB {ub}")
        start = time.time()

        enc = encoding_signed_bipartite.TwinWidthEncoding()
        cb = enc.run(g, slv.Cadical103, ub)
    else:
        g = parser.parse(instance_file_name)[0]
        result = process_graph(g,file_name)
    if output_graphs:
            instance_name = os.path.split(instance_file_name)[-1]
            mg = cb[2]
            for u, v in g.edges:
                g[u][v]["red"] = False

            for i, n in enumerate(cb[1]):
                if n not in mg:
                    t = None
                    n = None
                else:
                    t = mg[n]
                with open(f"{instance_name}_{i}.dot", "w") as f:
                    f.write(tools.dot_export(g, n, t, True))
                with open(f"{instance_name}_{i}.png", "w") as f:
                    subprocess.run(["dot", "-Tpng",
                        f"{instance_name}_{i}.dot"], stdout=f)

                if n is None:
                    break

                tns = set(g.successors(t))
                tnp = set(g.predecessors(t))
                nns = set(g.successors(n))
                nnp = set(g.predecessors(n))

                nn = nns | nnp
                tn = tns | tnp

                for v in nn:
                    if v != t:
                        # Red remains, should edge exist
                        if (v in g[n] and g[n][v]['red']) \
                            or v not in tn or (v in nns and v not in tns) or (
                                v in nnp and v not in tnp):
                            if g.has_edge(t, v):
                                g[t][v]['red'] = True
                            elif g.has_edge(v, t):
                                g[v][t]['red'] = True
                            else:
                                g.add_edge(t, v, red=True)

                for v in tn:
                    if v not in nn:
                        if g.has_edge(t, v):
                            g[t][v]['red'] = True
                        elif g.has_edge(v, t):
                            g[v][t]['red'] = True
                        else:
                            g.add_edge(v, t, red=True)

                for u in list(g.successors(n)):
                    g.remove_edge(n, u)
                for u in list(g.predecessors(n)):
                    g.remove_edge(u, n)
                g.nodes[n]["del"] = True

    if save_result_to_csv:
        df = DataFrame().from_records([result])
        results_file_name = (BASE_PATH /
            f"results_tww_{datetime.now()}.csv").resolve()
        df.to_csv(results_file_name)
    if save_pace_output:
        pace_output_file_name =  (BASE_PATH / (str(file_name).split(".")[0]+"_pace_output.gr")).resolve().as_posix()
        print_contraction_tree(result["contraction_tree"],
            result["elimination_ordering"],
            result["tww"], g.number_of_nodes(),True,pace_output_file_name)
    return result

def delete_files_starting_with(prefix):
    """
    Delete all files in the current directory that start with the given prefix.

    Args:
        prefix (str): The string that the files to be deleted should start with.
    """
    current_directory = os.getcwd()

    for filename in os.listdir(current_directory):
        if filename.startswith(prefix):
            file_path = os.path.join(current_directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted: {file_path}")
            else:
                logger.debug(f"Skipping non-file: {file_path}")



if __name__ == "__main__":
    app()
