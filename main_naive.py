import sys

import networkx as nx
import matplotlib.pyplot as plt

from twin_width import heuristic
import pace_parser
from naive import get_naive
from logger import logger

instance = sys.argv[-1]
logger.debug(instance)

g = pace_parser.parse(instance)

nx.draw(g)
plt.savefig("graph_visu.png")


logger.debug(f"{len(g.nodes)} {len(g.edges)}")

ub = heuristic.get_ub(g)
ub2 = heuristic.get_ub2(g)
logger.debug(f"UB {ub} {ub2}")
logger.debug(get_naive(g))
