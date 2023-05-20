import networkx as nx
import bz2
import sys

def parse(path):
    if path.lower().endswith(".bz2"):
        f = bz2.open(path, mode='rb')
    else:
        f = open(path)

    g = nx.Graph()
    c_vertices = set()

    mode_edges = True
    for line in f:
        try:
            line = line.decode('ascii')
        except AttributeError:
            pass
        entries = line.strip().split()
        if mode_edges:
            if line.lower().strip() == "cvertices":
                mode_edges = False
            else:
                if len(entries) == 2 or (len(entries) == 3 and entries[0].lower().strip() == "e"):
                    try:
                        g.add_edge(int(entries[-2].strip()), int(entries[-1].strip()))
                    except ValueError:
                        if entries[0].lower().strip() == "e":
                            g.add_edge(entries[-2].strip(), entries[-1])
        else:
            if len(entries) == 1:
                try:
                    c_vertices.add(int(entries[0]))
                except ValueError:
                    pass
    f.close()

    return g, c_vertices


def parse_cnf(path):
    if path.lower().endswith(".bz2"):
        f = bz2.open(path, mode='rb')
    else:
        f = open(path)

    g = nx.DiGraph()

    clause = 1
    for cl in f:
        try:
            cl = cl.decode('ascii')
        except AttributeError:
            pass

        if not cl.startswith("p") and not cl.startswith("c") and not cl.startswith("%"):
            if len(cl.strip()) != 0:
                fields = [int(x.strip()) for x in cl.strip().split(" ") if x.strip() != "0"]
                if len(fields) > 0:
                    for cf in fields:
                        if cf > 0:
                            g.add_edge(f"c{clause}", f"v{abs(cf)}")
                        else:
                            g.add_edge(f"v{abs(cf)}", f"c{clause}")
                    clause += 1

    return g

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
                    
                
                
        
        
        
    