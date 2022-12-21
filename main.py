#!/usr/bin/env python
# -*- coding: utf-8 -*-
import heuristic
import pace_parser
from naive import get_naive
import sys
import networkx as nx
import matplotlib.pyplot as plt
from tools import prime_paley


instance = sys.argv[-1]
print(instance)

g = pace_parser.parse(instance)
# g = prime_paley(8)

nx.draw(g)
plt.savefig("graph_visu.png")


print(f"{len(g.nodes)} {len(g.edges)}")

ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
print(f"UB {ub} {ub2}")
print(get_naive(g))
