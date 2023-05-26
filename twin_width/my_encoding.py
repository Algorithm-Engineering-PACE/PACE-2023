import logging
import time
from typing import List, Dict, Optional

from networkx import Graph
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from threading import Timer
import twin_width.formula_ops as fops
# import networkx as nx
# import matplotlib.pyplot as plt


class MyTwinWidthEncoding:
    def __init__(self, g, d, card_enc: EncType=1):
        self.edge: Optional[List[Dict[int, int]]] = None
        self.merge: Optional[Dict] = None
        self.node_map: Optional[Dict] = None
        self.pool: Optional[IDPool] = None
        self.formula: Optional[CNF] = None
        self.totalizer: Optional[EncType] = None
        self.g = g
        self.card_enc = card_enc

        self.num_original_vertices = len(g.nodes)
        self.num_total_vertices = 2 * len(g.nodes) - d - 2
        self._parent_start_index = len(g.nodes) + 1
        self.child_end_index = 2 * len(g.nodes) - d - 3

    def update_num_vertices(self, g, d):
        self.num_total_vertices = (2 * len(g.nodes)) - d - 2
        self.child_end_index = (2 * len(g.nodes)) - d - 3


    def remap_graph(self, g):
        self.node_map = {}
        cnt = 1
        gn = Graph()

        for u, v in g.edges():
            if u not in self.node_map:
                self.node_map[u] = cnt
                cnt += 1
            if v not in self.node_map:
                self.node_map[v] = cnt
                cnt += 1

            gn.add_edge(self.node_map[u], self.node_map[v])

        return gn

    def init_var(self, g, d):
        self.edge = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.red = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.left_child = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.right_child = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.vanished = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.left_edge = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.right_edge = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.left_red = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.right_red = [{} for _ in range(0, self.num_total_vertices + 1)]
        self.red_unvanished = [
            [{} for _ in range(0, self.num_total_vertices + 1)]
            for _ in range(0, self.num_total_vertices + 1)
        ]

        # variables for edges and red-edges
        # we have edge_i_j and red_i_j for i<j
        for i in range(1, self.num_total_vertices + 1):
            for j in range(i + 1, self.num_total_vertices + 1):
                self.edge[i][j] = self.pool.id(f"edge_{i}_{j}")
                self.red[i][j] = self.pool.id(f"red_{i}_{j}")

        for i in range(self._parent_start_index, self.num_total_vertices + 1):
            for j in range(1, i + 1):
                # variables for parent-child relation
                self.left_child[i][j] = self.pool.id(f"left_child_{i}_{j}")
                self.right_child[i][j] = self.pool.id(f"right_child_{i}_{j}")

                # variables denoting whether v_j is already contracted at the
                # time point after v_i is formed
                self.vanished[i][j] = self.pool.id(f"vanished_{i}_{j}")

                # variables encoding the adjacencies and red-adjacencies of
                # the left and right children
                # as in edge and red they are defined for _i_j such that i<j
                self.left_edge[j][i] = self.pool.id(f"left_edge_{j}_{i}")
                self.right_edge[j][i] = self.pool.id(f"right_edge_{j}_{i}")
                self.left_red[j][i] = self.pool.id(f"left_red_{j}_{i}")
                self.right_red[j][i] = self.pool.id(f"right_red_{j}_{i}")

        for i in range(self._parent_start_index, self.num_total_vertices + 1):
            for j in range(1, i + 1):
                for k in range(1, i + 1):
                    if i == j:
                        continue
                    a = min(j, k)
                    b = max(j, k)
                    self.red_unvanished[i][a][b] = self.pool.id(
                        f"right_unvanished_{i}_{a}_{b}"
                    )

    def max_parent_start_index(self, c):
        return max(self._parent_start_index, c + 1)

    # encoding the original edges of the graph and
    # initializing initial red edges to be empty
    def encode_original_edges(self, g):
        for i in range(1, self.num_original_vertices + 1):
            nb = set(g.neighbors(i))
            for j in range(i + 1, self.num_original_vertices + 1):
                if j in nb:
                    self.formula.append([self.edge[i][j]])
                else:
                    self.formula.append([-self.edge[i][j]])
                self.formula.append([-self.red[i][j]])

    # clauses encoding the parent-child relation
    # every child has at most one parent
    # every non-original vertex has exactly one left child
    # self.left_child[i][j] is true if and only if v_j is contraccted to a new vertex v_i. for i>n.
    # and one right child
    def encode_parent_child(self):
        # every child has at most one parent
        for c in range(1, self.child_end_index+1):
            parent_left_list = [
                self.left_child[p][c]
                for p in range(self.max_parent_start_index(c), self.num_total_vertices + 1)
            ]
            parent_right_list = [
                self.right_child[p][c]
                for p in range(self.max_parent_start_index(c), self.num_total_vertices + 1)
            ]
            parent_list = parent_left_list + parent_right_list
            self.formula.extend(CardEnc.atmost(parent_list, vpool=self.pool, bound=1))

        # every non-original vertex has exactly one left child
        # and one right child
        for p in range(self._parent_start_index, self.num_total_vertices + 1):
            self.formula.extend(
                CardEnc.equals(
                    [self.left_child[p][c] for c in range(1, p)],
                    vpool=self.pool,
                    encoding=self.card_enc,
                )
            )
            self.formula.extend(
                CardEnc.equals(
                    [self.right_child[p][c] for c in range(1, p)],
                    vpool=self.pool,
                    encoding=self.card_enc,
                )
            )

    # encoding whether j is vanished after creating i
    # j is vanished if i is the parent of j
    # j is also vanished if j was vanished after creating i-1
    def encode_vanished(self):
        s = self._parent_start_index

        for i in range(s, self.num_total_vertices + 1):
            self.formula.append([-self.vanished[i][i]])

        # first encode when i is the first non-original vertex
        # the two vertices merging into it gets vanished
        for j in range(1, s):
            self.formula.extend(fops.equiv(
                self.vanished[s][j],
                [self.left_child[s][j], self.right_child[s][j]]
            ))
        # now encode it for the remaining non-original vertices
        # the two vertices merging into i gets vanished
        # and also the ones that are vanished in i-1 remain vanished
        for i in range(s + 1, self.num_total_vertices + 1):
            for j in range(1, i):
                self.formula.extend(fops.equiv(
                    self.vanished[i][j],
                    [
                        self.vanished[i - 1][j],
                        self.left_child[i][j],
                        self.right_child[i][j]
                    ]
                ))


    # encoding edges and red-edges for contracted vertices
    def encode_edges(self):
        for p in range(self._parent_start_index, self.num_total_vertices + 1):
            for c in range(1, p):
                for i in range(1, p):
                    if i == c:
                        continue
                    a = min(i, c)
                    b = max(i, c)

                    self.formula.extend(fops.implies(
                        fops.conj([self.left_child[p][c]], [self.edge[a][b]]),
                        [self.left_edge[i][p]],
                    ))

                    self.formula.extend(fops.implies(
                        fops.conj([self.right_child[p][c]], [self.edge[a][b]]),
                        [self.right_edge[i][p]],
                    ))

                    self.formula.extend(fops.implies(
                        fops.conj([self.left_child[p][c]], [self.red[a][b]]),
                        [self.left_red[i][p]],
                    ))

                    self.formula.extend(fops.implies(
                        fops.conj([self.right_child[p][c]], [self.red[a][b]]),
                        [self.right_red[i][p]],
                    ))

                    self.formula.extend(fops.implies(
                        fops.conj([self.left_child[p][c]], [-self.edge[a][b]]),
                        [-self.left_edge[i][p]],
                    ))

                    self.formula.extend(fops.implies(
                        fops.conj([self.right_child[p][c]], [-self.edge[a][b]]),
                        [-self.right_edge[i][p]],
                    ))

                    self.formula.extend(fops.implies(
                        fops.conj([self.left_child[p][c]], [-self.red[a][b]]),
                        [-self.left_red[i][p]],
                    ))

                    self.formula.extend(fops.implies(
                        fops.conj([self.right_child[p][c]], [-self.red[a][b]]),
                        [-self.right_red[i][p]],
                    ))
        # Encoding edges for new vertices
        for i in range(self._parent_start_index, self.num_total_vertices + 1):
            for j in range(1, i):
                self.formula.extend([[-self.edge[j][i], self.left_edge[j][i], self.right_edge[j][i]],
                                     [-self.vanished[i][j], -self.edge[j][i]],
                                     [-self.left_edge[j][i], self.edge[j][i],self.vanished[i][j]],
                                     [-self.right_edge[j][i], self.edge[j][i],self.vanished[i][j]]
                ])

                self.formula.extend([[-self.red[j][i], self.edge[j][i]],
                                     [-self.red[j][i], self.left_red[j][i], self.right_red[j][i],
                                         self.left_edge[j][i], self.right_edge[j][i]],
                                     [-self.red[j][i], self.left_red[j][i], self.right_red[j][i],
                                         -self.left_edge[j][i], -self.right_edge[j][i]]
                ])

                self.formula.extend([[self.red[j][i], -self.edge[j][i], -self.left_red[j][i]],
                                     [self.red[j][i], -self.edge[j][i], -self.right_red[j][i]],
                                     [self.red[j][i], -self.edge[j][i], -self.left_edge[j][i],
                                         self.right_edge[j][i]],
                                     [self.red[j][i], -self.edge[j][i], -self.right_edge[j][i],
                                         self.left_edge[j][i]]
                ])



    def encode_red_unvanished(self):
        for i in range(self._parent_start_index, self.num_total_vertices + 1):
            for j in range(1, i + 1):
                for k in range(j + 1, i + 1):
                    self.formula.extend([[-self.vanished[i][j], -self.red_unvanished[i][j][k]],
                                         [-self.vanished[i][k], -self.red_unvanished[i][j][k]],
                                         [self.red[j][k], -self.red_unvanished[i][j][k]]]
                    )
                    self.formula.extend([[self.vanished[i][j], self.vanished[i][k],
                                          self.red_unvanished[i][j][k], -self.red[j][k]]])

    def encode_counters(self, d):
        for i in range(self._parent_start_index, self.num_total_vertices + 1):
            for j in range(1, i + 1):
                vars_to_count = [
                    self.red_unvanished[i][min(j, k)][max(j, k)]
                    for k in range(1, i + 1)
                    if j != k
                ]
                self.formula.extend(
                    CardEnc.atmost(
                        vars_to_count, bound=d, vpool=self.pool, encoding=EncType.totalizer
                    )
                )

    def encode_break_symmetry(self):
        for i in range(self._parent_start_index, self.num_total_vertices + 1):
            for j in range(1, i):
                for k in range(1, j):
                    self.formula.append([-self.left_child[i][j], -self.right_child[i][k]])

    def encode(self, g, d):
        self.update_num_vertices(g, d)
        g = self.remap_graph(g)
        #nx.draw(g, with_labels=True)
        #plt.savefig('graph.png')

        self.pool = IDPool()
        self.formula = CNF()
        self.init_var(g, d)
        self.encode_original_edges(g)
        self.encode_parent_child()
        self.encode_vanished()
        self.encode_edges()
        self.encode_red_unvanished()
        self.encode_counters(d)
        self.encode_break_symmetry()
        print(f"{len(self.formula.clauses)} / {self.formula.nv}")
        return self.formula

    def run(self, g, solver, start_bound, verbose=True, check=True, timeout=0):
        start = time.time()
        cb = start_bound

        if verbose:
            print(f"Created encoding in {time.time() - start}")

        done = []
        c_slv = None

        def interrupt():
            if c_slv is not None:
                c_slv.interrupt()
            done.append(True)

        timer = None
        if timeout > 0:
            timer = Timer(timeout, interrupt)
            timer.start()

        i = start_bound
        while i > 0:
            if done:
                break
            with solver() as slv:
                c_slv = slv
                formula = self.encode(g, i)
                slv.append_formula(formula)

                if done:
                    break

                if slv.solve() if timeout == 0 else slv.solve_limited():
                    model = slv.get_model()
                    elim_order = self.elim_order(model, i) # TODO: find elim order only for the last d
                    if verbose:
                        print(f"Found {i}")
                    i = cb =  cb - 1
                else:
                    cb += 1
                    if verbose:
                        print(f"Failed {i}")
                    break

                if verbose:
                    print(f"Finished cycle in {time.time() - start}")
        if timer is not None:
            timer.cancel()
        if cb == 0:  # note that twin-width 0 case is taken care of seperately in main
            cb = 1
        return cb, elim_order, None, time.time() - start
    
    def elim_order(self, model, d):
        model = {abs(x): x > 0 for x in model}
        num_parents = self.num_total_vertices-self.num_original_vertices
        node_id = [*range(self.num_original_vertices+1)]+[0]*(num_parents)
        children_ids = [(0, 0)]*(self.num_total_vertices+2)
        parent = [0]*(self.num_total_vertices+2)

        for i in range(self._parent_start_index, self.num_total_vertices+1):
            left = right = 0
            for j in range(1, i):
                if model[self.left_child[i][j]]:
                    if(left > 0):
                        print(f"Two left childs for {i}")
                        return None
                    left = j
                    if parent[j]:
                        print(f"Two parents for {j} namely {parent[j]} and {i}")
                        return None
                    parent[j] = i
                if model[self.right_child[i][j]]:
                    if(right > 0):
                        print(f"Two right childs for {i}")
                        return None
                    right = j
                    if parent[j]:
                        print(f"Two parents for {j} namely {parent[j]} and {i}")
                        print(f"left child of {i} is {left} and right child of {i} is {right}")
                        return None
                    parent[j] = i
            if left == 0:
                print(f"no left child for {i} ")
                return None
            if right == 0:
                print(f"no right child for {i} ")
                return None
            children_ids[i] = (node_id[left], node_id[right])
            node_id[i] = node_id[left]
        roots_ids = [node_id[i] for i in range(1, self.num_total_vertices+1) if not parent[i]]
        if(len(roots_ids) != d+2):
            print(f"Number of uncontracted vertices is {len(roots_ids)} but expected {d+2}")
            print(f"parents:{parent}")
            return None
        logging.debug(f"uncontracted vertices ids: {roots_ids}")
        root1_id = roots_ids[0]
        final_contractions = []
        for i in range(1, len(roots_ids)):
            final_contractions.append((root1_id, roots_ids[i]))
        return children_ids[self._parent_start_index:self.num_total_vertices] + final_contractions




    def decode(self, model, g, d):
        g = g.copy()
        model = {abs(x): x > 0 for x in model}
        unmap = {}
        for u, v in self.node_map.items():
            unmap[v] = u

        # Find merge targets and elimination order
        mg = {}
        od = []

        for i in range(1, len(g.nodes) + 1):
            for j in range(1, len(g.nodes) + 1):
                if model[self.ord[i][j]]:
                    if len(od) >= i:
                        print("Double order")
                    od.append(j)
            if len(od) < i:
                print("Order missing")
        if len(set(od)) < len(od):
            print("Node twice in order")

        for i in range(1, len(g.nodes) + 1 - d):
            for j in range(i + 1, len(g.nodes) + 1):
                if model[self.merge[i][j]]:
                    if od[i - 1] in mg:
                        print("Error, double merge!")
                    mg[od[i - 1]] = od[j - 1]

        # Check edges relation...
        for i in range(0, len(g.nodes) - d):
            for j in range(i + 1, len(g.nodes) - d):
                if model[self.edge[i + 1][j + 1]] ^ g.has_edge(
                    unmap[od[i]], unmap[od[j]]
                ):
                    if model[self.edge[i + 1][j + 1]]:
                        print(
                            f"Edge error: Unknown edge in model {i+1}, {j+1} = {od[i], od[j]}"
                        )
                    else:
                        print("Edge error: Edge not in model")

        # Perform contractions, last node needs not be contracted...
        for u, v in g.edges:
            g[u][v]["red"] = False

        c_max = 0
        step = 1
        for n in od[:-d]:
            t = unmap[mg[n]]
            n = unmap[n]
            tn = set(g.neighbors(t))
            tn.discard(n)
            nn = set(g.neighbors(n))

            for v in nn:
                if v != t:
                    # Red remains, should edge exist
                    if v in tn and g[n][v]["red"]:
                        g[t][v]["red"] = True
                    # Add non-existing edges
                    if v not in tn:
                        g.add_edge(t, v, red=True)
            for v in tn:
                if v not in nn:
                    g[t][v]["red"] = True
            g.remove_node(n)

            # Count reds...
            for u in g.nodes:
                cc = 0
                for v in g.neighbors(u):
                    if g[u][v]["red"]:
                        cc += 1
                        u2, v2 = (
                            od.index(self.node_map[u]) + 1,
                            od.index(self.node_map[v]) + 1,
                        )
                        u2, v2 = min(u2, v2), max(u2, v2)
                        if not model[self.red[step][u2][v2]]:
                            print(f"Missing red edge in step {step}")
                if cc > d:
                    print(f"Exceeded bound in step {step}")
                c_max = max(c_max, cc)

            step += 1
        print(f"Done {c_max}/{d}")

        return c_max
