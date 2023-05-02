import os
import subprocess
import sys
import time
from threading import Timer
from datetime import datetime
import pysat.solvers as slv

import twin_width.encoding as encoding
import twin_width.encoding5 as other
import twin_width.encoding6 as other2
import twin_width.encoding_signed_bipartite as encoding_signed_bipartite
import twin_width.heuristic as heuristic
import twin_width.parser as parser
import preprocessing
import tools
from pathlib import Path
from pandas import DataFrame


BASE_PATH = Path(__file__).parent


INSTANCES_PATH = (BASE_PATH / "tiny-set").resolve() 

if not os.path.exists(INSTANCES_PATH):
    print(f"folder is not exists {INSTANCES_PATH}")
files = sorted(os.listdir(INSTANCES_PATH))
graph_files = filter(lambda file_name: file_name.endswith(".gr"), files)
results = []
for file_name in files:
    instance_file_name = (INSTANCES_PATH / file_name).resolve().as_posix() 

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

        print(f"{len(g.nodes)} {len(g.edges)}")
       
        ## our preprocessing 
        g = preprocessing.preproccess(g)

        if len(g.nodes) == 1:
            print("Done, width: 0")
            exit(0)

        ub = heuristic.get_ub(g)
        ub2 = heuristic.get_ub2(g)
        print(f"UB {ub} {ub2}")
        ub = min(ub, ub2)

        start = time.time()
        enc = encoding.TwinWidthEncoding()
        cb = enc.run(g, slv.Cadical103, ub)
    
    duration = time.time() - start 
    print(f"Finished, result: {cb}")
    results.append({"instance_name": file_name
                    ,"# nodes": g.number_of_nodes()
                    ,"# edges": g.number_of_edges()
                    ,"tww": cb[0]
                    ,"elimination_ordering": cb[1]
                    ,"contraction_tree": cb[2]
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
                    subprocess.run(["dot", "-Tpng", f"{instance_name}_{i}.dot"], stdout=f)

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
                        if (v in g[n] and g[n][v]['red']) or v not in tn or (v in nns and v not in tns) or (
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
results_file_name = (BASE_PATH / f"results_tww_{datetime.now().date()}.csv").resolve()
df.to_csv(results_file_name)    