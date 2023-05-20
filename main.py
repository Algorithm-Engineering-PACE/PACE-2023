import os
import subprocess
import sys
import time
from datetime import datetime
import pysat.solvers as slv

import twin_width.my_encoding as encoding
import twin_width.encoding_signed_bipartite as encoding_signed_bipartite
import twin_width.heuristic as heuristic
import twin_width.parser as parser
import preprocessing
import tools
from pathlib import Path
from pandas import DataFrame
import typer

app = typer.Typer()
BASE_PATH = Path(__file__).parent

@app.command()
def clean_results():
    delete_files_starting_with("results_tww")

@app.command()
def process_graphs_from_dir(instance_path: Path, start: int =0, to: int=-1):

    instance_path = (BASE_PATH / instance_path).resolve()

    if not os.path.exists(instance_path):
        print(f"folder is not exists {instance_path}")
    files = sorted(os.listdir(instance_path))
    files = filter(lambda file_name: file_name.endswith(".gr"), files)
    results = []
    for file_name in files:
        process_file(instance_path, file_name, results=results)

@app.command()
def process_graph_from_instance(file_name: Path):
    process_file(BASE_PATH, file_name)

def process_file(instance_path: Path, file_name: str | Path,
    csv_time: datetime=datetime.now(), results: list=[]):
    instance_file_name = (instance_path / file_name).resolve().as_posix()

    print(f"processing file {file_name}....")
    output_graphs = False
    if any(x == "-l" for x in sys.argv[1:-1]):
        output_graphs = True

    if instance_file_name.endswith(".cnf"):
        g = parser.parse_cnf(file_name)
        ub = heuristic.get_ub2_polarity(g)

        print(f"UB {ub}")
        start = time.time()

        enc = encoding_signed_bipartite.TwinWidthEncoding()
        cb = enc.run(g, slv.Cadical103, ub)
    else:
        g = parser.parse(instance_file_name)[0]

        ## our preprocessing
        g = preprocessing.preproccess(g)

        if len(g.nodes) <= 1:
            print("Done, width: 0")
            results.append({"instance_name": file_name
                            ,"# nodes": g.number_of_nodes()
                            ,"# edges": g.number_of_edges()
                            ,"tww": 0
                            ,"elimination_ordering": None
                            ,"contraction_tree": None
                            ,"cycle_times": None
                            ,"duration": None
                                    })
            return

        ub = heuristic.get_ub(g)
        ub2 = heuristic.get_ub2(g)
        print(f"UB {ub} {ub2}")
        ub = min(ub, ub2)

        start = time.time()
        enc = encoding.MyTwinWidthEncoding(g, ub)
        cb = enc.run(g, slv.Cadical103, ub)

    duration = time.time() - start
    print(f"Finished, result: {cb}")
    results.append({"instance_name": file_name
                    ,"# nodes": g.number_of_nodes()
                    ,"# edges": g.number_of_edges()
                    ,"tww": cb[0]
                    ,"elimination_ordering": cb[1]
                    ,"contraction_tree": cb[2]
                    ,"cycle_times": cb[3]
                    ,"duration": duration
                            })

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

    df =  DataFrame().from_records(results)
    results_file_name = (BASE_PATH /
        f"results_tww_{csv_time}.csv").resolve()
    df.to_csv(results_file_name)

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
                print(f"Deleted: {file_path}")
            else:
                print(f"Skipping non-file: {file_path}")



if __name__ == "__main__":
    app()
