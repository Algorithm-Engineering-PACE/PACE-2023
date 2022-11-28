#!/usr/bin/env python
# -*- coding: utf-8 -*-
import heuristic
import parser
import sys

instance = sys.argv[-1]
print(instance)

g = parser.parse(instance)[0]
    # g = tools.prime_paley(29)

print(f"{len(g.nodes)} {len(g.edges)}")

ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
print(f"UB {ub} {ub2}")
ub = min(ub, ub2)
