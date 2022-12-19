from operator import itemgetter
from itertools import combinations

from heuristic import get_ub, get_open_neighberhood_without_node

def get_naive(g):
    ub = get_ub(g)
    g = g.copy()
    for u, v in g.edges:
        g[u][v]['red'] = False

    return helper(g, ub)

def contract(g_in, u, v):
    g = g_in.copy()
    n = u  # u
    t = v  # v

    tn = get_open_neighberhood_without_node(g.neighbors(t), n)
    nn = set(g.neighbors(n))
    
    for v in nn:  # for every neighbour  of u
        if v != t:
            # Red remains, should edge exist
            if v in tn and g[n][v]['red']:
                g[t][v]['red'] = True
            # Add non-existing edges
            if v not in tn:
                g.add_edge(t, v, red=True)

    for v in tn:
        if v not in nn:
            g[t][v]['red'] = True

    g.remove_node(n)
    return g

def get_red_degree(g):
    # Find d for this d-trigraph
    c_max = 0
    for u in g.nodes:
        cc = 0
        for v in g.neighbors(u):
            if g[u][v]['red']:
                cc += 1
        c_max = max(c_max, cc)

    return c_max




def helper(g, ub):
    if len(g.nodes) == 1:
        return 0, {}, []

    res = []
    g_in = g.copy()
    for u,v in combinations(g_in.nodes, 2):
        g = contract(g_in, u, v)
        c_max = get_red_degree(g)

        if c_max >= ub:
            print(f"Not exploring due to ub the following edge in search tree {u} - {v},{c_max} left nodes: {len(g.nodes)}")
            continue  # Optimization

        d, mg, od = helper(g, ub)
        if d >= 0:
            od.append(u)
            mg[u] = v
            d = max(c_max, d)
            res.append((d, mg, od))

    if len(res) == 0:
        return -1, {}, []
    else:
        return min(res, key=itemgetter(0))

 
