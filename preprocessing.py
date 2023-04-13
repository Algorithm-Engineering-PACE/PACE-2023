from enum import Enum
import networkx as nx
from collections import Set

## sagemath implementation for graph modular decomposition 

class NodeType(Enum):
    """
    NodeType is an enumeration class used to define the various types of nodes
    in modular decomposition tree.

    The various node types defined are

    - ``PARALLEL`` -- indicates the node is a parallel module

    - ``SERIES`` -- indicates the node is a series module

    - ``PRIME`` -- indicates the node is a prime module

    - ``FOREST`` -- indicates a forest containing trees

    - ``NORMAL`` -- indicates the node is normal containing a vertex
    """
    PRIME = 0
    SERIES = 1
    PARALLEL = 2
    NORMAL = 3
    FOREST = -1

    def __repr__(self):
        r"""
        String representation of this node type.

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import NodeType
            sage: repr(NodeType.PARALLEL)
            'PARALLEL'
        """
        return self.name

    def __str__(self):
        """
        String representation of this node type.

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import NodeType
            sage: str(NodeType.PARALLEL)
            'PARALLEL'
        """
        return repr(self)


class NodeSplit(Enum):
    """
    Enumeration class used to specify the split that has occurred at the node or
    at any of its descendants.

    ``NodeSplit`` is defined for every node in modular decomposition tree and is
    required during the refinement and promotion phase of modular decomposition
    tree computation. Various node splits defined are

    - ``LEFT_SPLIT`` -- indicates a left split has occurred

    - ``RIGHT_SPLIT`` -- indicates a right split has occurred

    - ``BOTH_SPLIT`` -- indicates both left and right split have occurred

    - ``NO_SPLIT`` -- indicates no split has occurred
    """
    LEFT_SPLIT = 1
    RIGHT_SPLIT = 2
    BOTH_SPLIT = 3
    NO_SPLIT = 0


class VertexPosition(Enum):
    """
    Enumeration class used to define position of a vertex w.r.t source in
    modular decomposition.

    For computing modular decomposition of connected graphs a source vertex is
    chosen. The position of vertex is w.r.t this source vertex. The various
    positions defined are

    - ``LEFT_OF_SOURCE`` -- indicates vertex is to left of source and is a
      neighbour of source vertex

    - ``RIGHT_OF_SOURCE`` -- indicates vertex is to right of source and is
      connected to but not a neighbour of source vertex

    - ``SOURCE`` -- indicates vertex is source vertex
    """
    LEFT_OF_SOURCE = -1
    RIGHT_OF_SOURCE = 1
    SOURCE = 0


class Node:
    """
    Node class stores information about the node type, node split and index of
    the node in the parent tree.

    Node type can be ``PRIME``, ``SERIES``, ``PARALLEL``, ``NORMAL`` or
    ``FOREST``. Node split can be ``NO_SPLIT``, ``LEFT_SPLIT``, ``RIGHT_SPLIT``
    or ``BOTH_SPLIT``. A node is split in the refinement phase and the split
    used is propagated to the ancestors.

    - ``node_type`` -- is of type NodeType and specifies the type of node

    - ``node_split`` -- is of type NodeSplit and specifies the type of splits
      which have occurred in the node and its descendants

    - ``index_in_root`` -- specifies the index of the node in the forest
      obtained after promotion phase

    - ``comp_num`` -- specifies the number given to nodes in a (co)component
      before refinement

    - ``is_separated`` -- specifies whether a split has occurred with the node
      as the root
    """
    def __init__(self, node_type):
        r"""
        Create a node with the given node type.

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import *
            sage: n = Node(NodeType.SERIES); n.node_type
            SERIES
            sage: n.children
            []
        """
        self.node_type = node_type
        self.node_split = NodeSplit.NO_SPLIT
        self.index_in_root = -1
        self.comp_num = -1
        self.is_separated = False
        self.children = []

    def set_node_split(self, node_split):
        """
        Add node_split to the node split of self.

        ``LEFT_SPLIT`` and ``RIGHT_SPLIT`` can exist together in ``self`` as
        ``BOTH_SPLIT``.

        INPUT:

        - ``node_split`` -- node_split to be added to self

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import *
            sage: node = Node(NodeType.PRIME)
            sage: node.set_node_split(NodeSplit.LEFT_SPLIT)
            sage: node.node_split == NodeSplit.LEFT_SPLIT
            True
            sage: node.set_node_split(NodeSplit.RIGHT_SPLIT)
            sage: node.node_split == NodeSplit.BOTH_SPLIT
            True
            sage: node = Node(NodeType.PRIME)
            sage: node.set_node_split(NodeSplit.BOTH_SPLIT)
            sage: node.node_split == NodeSplit.BOTH_SPLIT
            True
        """
        if self.node_split == NodeSplit.NO_SPLIT:
            self.node_split = node_split
        elif ((self.node_split == NodeSplit.LEFT_SPLIT and
               node_split == NodeSplit.RIGHT_SPLIT) or
              (self.node_split == NodeSplit.RIGHT_SPLIT and
               node_split == NodeSplit.LEFT_SPLIT)):
            self.node_split = NodeSplit.BOTH_SPLIT

    def has_left_split(self):
        """
        Check whether ``self`` has ``LEFT_SPLIT``.

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import *
            sage: node = Node(NodeType.PRIME)
            sage: node.set_node_split(NodeSplit.LEFT_SPLIT)
            sage: node.has_left_split()
            True
            sage: node = Node(NodeType.PRIME)
            sage: node.set_node_split(NodeSplit.BOTH_SPLIT)
            sage: node.has_left_split()
            True
        """
        return (self.node_split == NodeSplit.LEFT_SPLIT or
                self.node_split == NodeSplit.BOTH_SPLIT)

    def has_right_split(self):
        """
        Check whether ``self`` has ``RIGHT_SPLIT``.

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import *
            sage: node = Node(NodeType.PRIME)
            sage: node.set_node_split(NodeSplit.RIGHT_SPLIT)
            sage: node.has_right_split()
            True
            sage: node = Node(NodeType.PRIME)
            sage: node.set_node_split(NodeSplit.BOTH_SPLIT)
            sage: node.has_right_split()
            True
        """
        return (self.node_split == NodeSplit.RIGHT_SPLIT or
                self.node_split == NodeSplit.BOTH_SPLIT)

    def __repr__(self):
        r"""
        Return a string representation of the node.

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import *
            sage: n = Node(NodeType.PRIME)
            sage: n.children.append(create_normal_node(1))
            sage: n.children.append(create_normal_node(2))
            sage: str(n)
            'PRIME [NORMAL [1], NORMAL [2]]'
        """
        if self.node_type == NodeType.SERIES:
            s = "SERIES "
        elif self.node_type == NodeType.PARALLEL:
            s = "PARALLEL "
        elif self.node_type == NodeType.PRIME:
            s = "PRIME "
        elif self.node_type == NodeType.FOREST:
            s = "FOREST "
        else:
            s = "NORMAL "

        s += str(self.children)
        return s

    def __eq__(self, other):
        r"""
        Compare two nodes for equality.

        EXAMPLES::

            sage: from sage.graphs.graph_decompositions.modular_decomposition import *
            sage: n1 = Node(NodeType.PRIME)
            sage: n2 = Node(NodeType.PRIME)
            sage: n3 = Node(NodeType.SERIES)
            sage: n1 == n2
            True
            sage: n1 == n3
            False
        """
        return (self.node_type == other.node_type and
                self.node_split == other.node_split and
                self.index_in_root == other.index_in_root and
                self.comp_num == other.comp_num and
                self.is_separated == other.is_separated and
                self.children == other.children)


def create_prime_node():
    """
    Return a prime node with no children

    OUTPUT:

    A node object with node_type set as NodeType.PRIME

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import create_prime_node
        sage: node = create_prime_node()
        sage: node
        PRIME []
    """
    return Node(NodeType.PRIME)


def create_parallel_node():
    """
    Return a parallel node with no children

    OUTPUT:

    A node object with node_type set as NodeType.PARALLEL

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import create_parallel_node
        sage: node = create_parallel_node()
        sage: node
        PARALLEL []
    """
    return Node(NodeType.PARALLEL)


def create_series_node():
    """
    Return a series node with no children

    OUTPUT:

    A node object with node_type set as NodeType.SERIES

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import create_series_node
        sage: node = create_series_node()
        sage: node
        SERIES []
    """
    return Node(NodeType.SERIES)


def create_normal_node(vertex):
    """
    Return a normal node with no children

    INPUT:

    - ``vertex`` -- vertex number

    OUTPUT:

    A node object representing the vertex with node_type set as NodeType.NORMAL

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import create_normal_node
        sage: node = create_normal_node(2)
        sage: node
        NORMAL [2]
    """
    node = Node(NodeType.NORMAL)
    node.children.append(vertex)
    return node


def print_md_tree(root):
    """
    Print the modular decomposition tree

    INPUT:

    - ``root`` -- root of the modular decomposition tree

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: print_md_tree(modular_decomposition(graphs.IcosahedralGraph()))
        PRIME
         1
         5
         7
         8
         11
         0
         2
         6
         3
         9
         4
         10
    """

    def recursive_print_md_tree(root, level):
        """
        Print the modular decomposition tree at root

        INPUT:

        - ``root`` -- root of the modular decomposition tree

        - ``level`` -- indicates the depth of root in the original modular
          decomposition tree
        """
        if root.node_type != NodeType.NORMAL:
            print("{}{}".format(level, str(root.node_type)))
            for tree in root.children:
                recursive_print_md_tree(tree, level + " ")
        else:
            print("{}{}".format(level, str(root.children[0])))

    recursive_print_md_tree(root, "")


def gamma_classes(graph):
    """
    Partition the edges of the graph into Gamma classes.

    Two distinct edges are Gamma related if they share a vertex but are not
    part of a triangle.  A Gamma class of edges is a collection of edges such
    that any edge in the class can be reached from any other by a chain of
    Gamma related edges (that are also in the class).

    The two important properties of the Gamma class

    * The vertex set corresponding to a Gamma class is a module
    * If the graph is not fragile (neither it or its complement is
      disconnected) then there is exactly one class that visits all the
      vertices of the graph, and this class consists of just the edges that
      connect the maximal strong modules of that graph.

    EXAMPLES:

    The gamma_classes of the octahedral graph are the three 4-cycles
    corresponding to the slices through the center of the octahedron::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import gamma_classes
        sage: g = graphs.OctahedralGraph()
        sage: sorted(gamma_classes(g), key=str)
        [frozenset({0, 1, 4, 5}), frozenset({0, 2, 3, 5}), frozenset({1, 2, 3, 4})]

    TESTS:

    Ensure that the returned vertex sets from some random graphs are modules::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import test_gamma_modules
        sage: test_gamma_modules(2, 10, 0.5)
    """
    from itertools import chain
    from networkx.utils.union_find import UnionFind

    edges = {frozenset(e) for e in graph.edges}
    pieces = UnionFind(edges)
    for v in graph.nodes:
        neighborhood = graph.subgraph(nx.neighbors(graph,v))
        neighborhood_com = nx.complement(neighborhood)
        connected_com_subgraph_list =  [graph.subgraph(c).copy() for c in nx.connected_components(neighborhood_com)]
        for component in connected_com_subgraph_list:
            v1 = list(component.nodes)[0]
            e = frozenset([v1, v])
            for vi in list(component.nodes)[1:]:
                ei = frozenset([vi, v])
                pieces.union(e, ei)
    res = {frozenset(chain.from_iterable(list(loe))): loe for loe in pieces.to_sets()}
    return res


def habib_maurer_algorithm(graph, g_classes=None):
    """
    Compute the modular decomposition by the algorithm of Habib and Maurer

    Compute the modular decomposition of the given graph by the algorithm of
    Habib and Maurer [HM1979]_ . If the graph is disconnected or its complement
    is disconnected return a tree with a ``PARALLEL`` or ``SERIES`` node at the
    root and children being the modular decomposition of the subgraphs induced
    by the components. Otherwise, the root is ``PRIME`` and the modules are
    identified by having identical neighborhoods in the gamma class that spans
    the vertices of the subgraph (exactly one is guaranteed to exist). The gamma
    classes only need to be computed once, as the algorithm computes the the
    classes for the current root and each of the submodules. See also [BM1983]_
    for an equivalent algorithm described in greater detail.

    INPUT:

    - ``graph`` -- the graph for which modular decomposition tree needs to be
      computed

    - ``g_classes`` -- dictionary (default: ``None``); a dictionary whose values
      are the gamma classes of the graph, and whose keys are a frozenset of the
      vertices corresponding to the class. Used internally.

    OUTPUT:

    The modular decomposition tree of the graph.

    EXAMPLES:

    The Icosahedral graph is Prime::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: print_md_tree(habib_maurer_algorithm(graphs.IcosahedralGraph()))
        PRIME
         1
         5
         7
         8
         11
         0
         2
         6
         3
         9
         4
         10

    The Octahedral graph is not Prime::

        sage: print_md_tree(habib_maurer_algorithm(graphs.OctahedralGraph()))
        SERIES
         PARALLEL
          0
          5
         PARALLEL
          1
          4
         PARALLEL
          2
          3

    Tetrahedral Graph is Series::

        sage: print_md_tree(habib_maurer_algorithm(graphs.TetrahedralGraph()))
        SERIES
         0
         1
         2
         3

    Modular Decomposition tree containing both parallel and series modules::

        sage: d = {2:[4,3,5], 1:[4,3,5], 5:[3,2,1,4], 3:[1,2,5], 4:[1,2,5]}
        sage: g = Graph(d)
        sage: print_md_tree(habib_maurer_algorithm(g))
        SERIES
         PARALLEL
          1
          2
         PARALLEL
          3
          4
         5

    Graph from Marc Tedder implementation of modular decomposition::

        sage: d = {1:[5,4,3,24,6,7,8,9,2,10,11,12,13,14,16,17], 2:[1],
        ....:       3:[24,9,1], 4:[5,24,9,1], 5:[4,24,9,1], 6:[7,8,9,1],
        ....:       7:[6,8,9,1], 8:[6,7,9,1], 9:[6,7,8,5,4,3,1], 10:[1],
        ....:       11:[12,1], 12:[11,1], 13:[14,16,17,1], 14:[13,17,1],
        ....:       16:[13,17,1], 17:[13,14,16,18,1], 18:[17], 24:[5,4,3,1]}
        sage: g = Graph(d)
        sage: test_modular_decomposition(habib_maurer_algorithm(g), g)
        True

    Graph from the :wikipedia:`Modular_decomposition`::

        sage: d2 = {1:[2,3,4], 2:[1,4,5,6,7], 3:[1,4,5,6,7], 4:[1,2,3,5,6,7],
        ....:       5:[2,3,4,6,7], 6:[2,3,4,5,8,9,10,11],
        ....:       7:[2,3,4,5,8,9,10,11], 8:[6,7,9,10,11], 9:[6,7,8,10,11],
        ....:       10:[6,7,8,9], 11:[6,7,8,9]}
        sage: g = Graph(d2)
        sage: test_modular_decomposition(habib_maurer_algorithm(g), g)
        True

    Tetrahedral Graph is Series::

        sage: print_md_tree(habib_maurer_algorithm(graphs.TetrahedralGraph()))
        SERIES
         0
         1
         2
         3

    Modular Decomposition tree containing both parallel and series modules::

        sage: d = {2:[4,3,5], 1:[4,3,5], 5:[3,2,1,4], 3:[1,2,5], 4:[1,2,5]}
        sage: g = Graph(d)
        sage: print_md_tree(habib_maurer_algorithm(g))
        SERIES
         PARALLEL
          1
          2
         PARALLEL
          3
          4
         5

    TESTS:

    Bad Input::

        sage: g = DiGraph()
        sage: habib_maurer_algorithm(g)
        Traceback (most recent call last):
        ...
        ValueError: Graph must be undirected

    Empty Graph is Prime::

        sage: g = Graph()
        sage: habib_maurer_algorithm(g)
        PRIME []


    Ensure that a random graph and an isomorphic graph have identical modular
    decompositions. ::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import permute_decomposition
        sage: permute_decomposition(2, habib_maurer_algorithm, 20, 0.5)
    """
    if graph.number_of_nodes() == 0:
        return create_prime_node()

    if graph.number_of_nodes() == 1:
        root = create_normal_node(next(graph.__iter__()))
        return root
    ## if g is not connected :
    ## 1.create parallel node 
    ## 2.set node's children to be g connected components (rec)  
    elif not nx.is_connected(graph):
        root = create_parallel_node()
        root.children = [habib_maurer_algorithm(graph.subgraph(component), g_classes)
                         for component in  nx.components.connected_components(graph)]
        return root
    ## if g is connected and g complement is connected :
    ## 1.create prime node 
    ## 2.get g modules using gamma_classes (does not contain all edges)
    ## 3.calc maximal modules of g 
    ## 4.set node's children to be g.subgraph(maximal module nodes) (rec)  
    g_comp = nx.complement(graph)
    if nx.is_connected(g_comp):
        from collections import defaultdict
        root = create_prime_node()
        if g_classes is None:
            g_classes = gamma_classes(graph)
        vertex_set = frozenset(graph)
        edges = [tuple(e) for e in g_classes[vertex_set]]
        sub = graph.edge_subgraph(edges=edges) ## todo: check this line
        d = defaultdict(list)
        for v in sub.nodes:
            for v1 in sub.neighbors(v):
                d[v1].append(v)
        ## vertices with the same adj list
        d1 = defaultdict(list)
        for k, v in d.items():
            d1[frozenset(v)].append(k)
        root.children = [habib_maurer_algorithm(graph.subgraph(sg), g_classes)
                         for sg in d1.values()]
        return root

    root = create_series_node()
    root.children = [habib_maurer_algorithm(graph.subgraph(component), g_classes)
                     for component in  nx.components.connected_components(g_comp)]
    return root

## sagemath tests for graph modular decomposition 


def get_module_type(graph):
    """
    Return the module type of the root of the modular decomposition tree of
    ``graph``.

    INPUT:

    - ``graph`` -- input sage graph

    OUTPUT:

    ``PRIME`` if graph is PRIME, ``PARALLEL`` if graph is PARALLEL and
    ``SERIES`` if graph is of type SERIES

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import get_module_type
        sage: g = graphs.HexahedralGraph()
        sage: get_module_type(g)
        PRIME
    """
    if not nx.is_connected(graph):
        return NodeType.PARALLEL
    elif nx.is_connected(nx.complement(graph)):
        return NodeType.PRIME
    return NodeType.SERIES

def form_module(index, other_index, tree_root, graph):
    r"""
    Forms a module out of the modules in the module pair.

    Let `M_1` and `M_2` be the input modules. Let `V` be the set of vertices in
    these modules. Suppose `x` is a neighbor of subset of the vertices in `V`
    but not all the vertices and `x` does not belong to `V`. Then the set of
    modules also include the module which contains `x`. This process is repeated
    until a module is formed and the formed module if subset of `V` is returned.

    INPUT:

    - ``index`` -- first module in the module pair

    - ``other_index`` -- second module in the module pair

    - ``tree_root`` -- modular decomposition tree which contains the modules
      in the module pair

    - ``graph`` -- graph whose modular decomposition tree is created

    OUTPUT:

    ``[module_formed, vertices]`` where ``module_formed`` is ``True`` if
    module is formed else ``False`` and ``vertices`` is a list of vertices
    included in the formed module

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: g = graphs.HexahedralGraph()
        sage: tree_root = modular_decomposition(g)
        sage: form_module(0, 2, tree_root, g)
        [False, {0, 1, 2, 3, 4, 5, 6, 7}]
    """
    vertices = set(get_vertices(tree_root.children[index]))
    vertices.update(get_vertices(tree_root.children[other_index]))

    # stores all neighbors which are common for all vertices in V
    common_neighbors = set()

    # stores all neighbors of vertices in V which are outside V
    all_neighbors = set()

    while True:
        # remove vertices from all_neighbors and common_neighbors
        all_neighbors.difference_update(vertices)
        common_neighbors.difference_update(vertices)

        for v in vertices:
            # stores the neighbors of v which are outside the set of vertices
            neighbor_list = set(graph.neighbors(v))
            neighbor_list.difference_update(vertices)

            # update all_neighbors and common_neighbors using the
            # neighbor_list
            all_neighbors.update(neighbor_list)
            common_neighbors.intersection_update(neighbor_list)

        if all_neighbors == common_neighbors:  # indicates a module is formed

            # module formed covers the entire graph
            if len(vertices) == graph.order():
                return [False, vertices]

            return [True, vertices]

        # add modules containing uncommon neighbors into the formed module
        for v in (all_neighbors - common_neighbors):
            for index in range(len(tree_root.children)):
                if v in get_vertices(tree_root.children[index]):
                    vertices.update(get_vertices(tree_root.children[index]))
                    break

def get_vertices(component_root):
    """
    Compute the list of vertices in the (co)component

    INPUT:

    - ``component_root`` -- root of the (co)component whose vertices need to be
      returned as a list

    OUTPUT:

    list of vertices in the (co)component

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: forest = Node(NodeType.FOREST)
        sage: forest.children = [create_normal_node(2),
        ....:                    create_normal_node(3), create_normal_node(1)]
        sage: series_node = Node(NodeType.SERIES)
        sage: series_node.children = [create_normal_node(4),
        ....:                         create_normal_node(5)]
        sage: parallel_node = Node(NodeType.PARALLEL)
        sage: parallel_node.children = [create_normal_node(6),
        ....:                           create_normal_node(7)]
        sage: forest.children.insert(1, series_node)
        sage: forest.children.insert(3, parallel_node)
        sage: get_vertices(forest)
        [2, 4, 5, 3, 6, 7, 1]
    """
    vertices = []

    # inner recursive function to recurse over the elements in the
    # ``component``
    def recurse_component(node, vertices):
        if node.node_type == NodeType.NORMAL:
            vertices.append(node.children[0])
            return
        for child in node.children:
            recurse_component(child, vertices)

    recurse_component(component_root, vertices)
    return vertices

def test_maximal_modules(tree_root, graph):
    r"""
    Test the maximal nature of modules in a modular decomposition tree.

    Suppose the module `M = [M_1, M_2, \cdots, n]` is the input modular
    decomposition tree. Algorithm forms pairs like `(M_1, M_2), (M_1, M_3),
    \cdots, (M_1, M_n)`; `(M_2, M_3), (M_2, M_4), \cdots, (M_2, M_n)`; `\cdots`
    and so on and tries to form a module using the pair. If the module formed
    has same type as `M` and is of type ``SERIES`` or ``PARALLEL`` then the
    formed module is not considered maximal. Otherwise it is considered maximal
    and `M` is not a modular decomposition tree.

    INPUT:

    - ``tree_root`` -- modular decomposition tree whose modules are tested for
      maximal nature

    - ``graph`` -- graph whose modular decomposition tree is tested

    OUTPUT:

    ``True`` if all modules at first level in the modular decomposition tree
    are maximal in nature

    EXAMPLES::

        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: g = graphs.HexahedralGraph()
        sage: test_maximal_modules(modular_decomposition(g), g)
        True
    """
    if tree_root.node_type != NodeType.NORMAL:
        for index, module in enumerate(tree_root.children):
            for other_index in range(index + 1, len(tree_root.children)):

                # compute the module formed using modules at index and
                # other_index
                module_formed = form_module(index, other_index,
                                            tree_root, graph)

                if module_formed[0]:
                    # Module formed and the parent of the formed module
                    # should not both be of type SERIES or PARALLEL
                    mod_type = get_module_type(graph.subgraph(module_formed[1]))
                    if (mod_type == tree_root.node_type and
                            (tree_root.node_type == NodeType.PARALLEL or
                             tree_root.node_type == NodeType.SERIES)):
                        continue
                    return False
    return True

def md_tree_to_graph(root):
    r"""
    Create a graph having the given MD tree.
    For the prime nodes we use that every path of length 4 or more is prime.
    TODO: accept a function that generates prime graphs as a parameter and
    use that in the prime nodes.
    EXAMPLES::
        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: tup1 = (NodeType.PRIME, 1, (NodeType.SERIES, 2, 3),
        ....:        (NodeType.PARALLEL, 4, 5), 6)
        sage: tree1 = nested_tuple_to_tree(tup1)
        sage: g1 = md_tree_to_graph(tree1)
        sage: g2 = Graph({1: [2, 3], 2: [1, 3, 4, 5], 3: [1, 2, 4, 5],
        ....:             4: [2, 3, 6], 5: [2, 3, 6], 6: [4, 5]})
        sage: g1.is_isomorphic(g2)
        True
    """
    from itertools import product, combinations

    def tree_to_vertices_and_edges(root):
        r"""
        Give the list of vertices and edges of the graph having the given md tree.
        """
        if root.node_type == NodeType.NORMAL:
            return (root.children, [])
        children_ve = [tree_to_vertices_and_edges(child) for child in root.children]
        vertices = [v for vs, es in children_ve for v in vs]
        edges = [e for vs, es in children_ve for e in es]
        vertex_lists = [vs for vs, es in children_ve]
        if root.node_type == NodeType.PRIME:
            for vs1, vs2 in zip(vertex_lists, vertex_lists[1:]):
                for v1, v2 in product(vs1, vs2):
                    edges.append((v1, v2))
        elif root.node_type == NodeType.SERIES:
            for vs1, vs2 in combinations(vertex_lists, 2):
                for v1, v2 in product(vs1, vs2):
                    edges.append((v1, v2))
        return (vertices, edges)

    vs, es = tree_to_vertices_and_edges(root)
    graph = nx.from_edgelist(es)
    graph.add_nodes_from(vs)
    return graph

def either_connected_or_not_connected(v, vertices_in_module, graph):
    """
    Check whether ``v`` is connected or disconnected to all vertices in the
    module.
    INPUT:
    - ``v`` -- vertex tested
    - ``vertices_in_module`` -- list containing vertices in the module
    - ``graph`` -- graph to which the vertices belong
    EXAMPLES::
        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: g = graphs.OctahedralGraph()
        sage: print_md_tree(modular_decomposition(g))
        SERIES
         PARALLEL
          0
          5
         PARALLEL
          1
          4
         PARALLEL
          2
          3
        sage: either_connected_or_not_connected(2, [1, 4], g)
        True
        sage: either_connected_or_not_connected(2, [3, 4], g)
        False
    """
    # marks whether vertex v is connected to first vertex in the module
    connected = graph.has_edge(vertices_in_module[0], v)

    # if connected is True then all vertices in module should be connected to
    # v else all should be disconnected
    return all(graph.has_edge(u, v) == connected for u in vertices_in_module)

def children_node_type(module, node_type):
    """
    Check whether the node type of the children of ``module`` is ``node_type``.
    INPUT:
    - ``module`` -- module which is tested
    - ``node_type`` -- input node_type
    EXAMPLES::
        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: g = graphs.OctahedralGraph()
        sage: tree_root = modular_decomposition(g)
        sage: print_md_tree(modular_decomposition(g))
        SERIES
         PARALLEL
          0
          5
         PARALLEL
          1
          4
         PARALLEL
          2
          3
        sage: children_node_type(tree_root, NodeType.SERIES)
        False
        sage: children_node_type(tree_root, NodeType.PARALLEL)
        True
    """
    return all(node.node_type == node_type for node in module.children)


def test_module(module, graph):
    """
    Test whether input module is actually a module
    INPUT:
    - ``module`` -- module which needs to be tested
    - ``graph`` -- input sage graph which contains the module
    OUTPUT:
    ``True`` if input module is a module by definition else ``False``
    EXAMPLES::
        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: g = graphs.HexahedralGraph()
        sage: tree_root = modular_decomposition(g)
        sage: test_module(tree_root, g)
        True
        sage: test_module(tree_root.children[0], g)
        True
    """
    # A single vertex is a module
    if module.node_type == NodeType.NORMAL:
        return True

    # vertices contained in module
    vertices_in_module = get_vertices(module)

    # vertices outside module
    vertices_outside = list(set(graph.nodes) - set(vertices_in_module))

    # Nested module with only one child
    if module.node_type != NodeType.NORMAL and len(module.children) == 1:
        return False

    # If children of SERIES module are all SERIES modules
    if module.node_type == NodeType.SERIES:
        if children_node_type(module, NodeType.SERIES):
            return False

    # If children of PARALLEL module are all PARALLEL modules
    if module.node_type == NodeType.PARALLEL:
        if children_node_type(module, NodeType.PARALLEL):
            return False

    # check the module by definition. Vertices in a module should all either
    # be connected or disconnected to any vertex outside module
    for v in vertices_outside:
        if not either_connected_or_not_connected(v, vertices_in_module, graph):
            return False
    return True

def test_modular_decomposition(tree_root, graph):
    """
    Test the input modular decomposition tree using recursion.
    INPUT:
    - ``tree_root`` -- root of the modular decomposition tree to be tested
    - ``graph`` -- graph whose modular decomposition tree needs to be tested
    OUTPUT:
    ``True`` if input tree is a modular decomposition else ``False``
    EXAMPLES::
        sage: from sage.graphs.graph_decompositions.modular_decomposition import *
        sage: g = graphs.HexahedralGraph()
        sage: test_modular_decomposition(modular_decomposition(g), g)
        True
    """
    if tree_root.node_type != NodeType.NORMAL:
        for module in tree_root.children:
            if not test_module(module, graph):
                # test whether modules pass the defining
                # characteristics of modules
                return False
            if not test_modular_decomposition(module,
                                              graph.subgraph(get_vertices(module))):
                # recursively test the modular decomposition subtrees
                return False

        if not test_maximal_modules(tree_root, graph):
            # test whether the mdoules are maximal in nature
            return False

    return True

## my functions

import pace_parser
from naive import get_naive
import networkx as nx
import matplotlib.pyplot as plt
from tools import prime_paley
import matplotlib.pyplot as plt
from pathlib import Path
import os
from shutil import copyfile



def is_prime(md_tree,graph):
    return md_tree.node_type == NodeType.PRIME and len(md_tree.children) == graph.number_of_nodes()


 
def create_graph_from_dm_tree(root,graph):
    decomposed_graph = nx.Graph()
    maximal_modules = {frozenset(get_vertices(child)) for child in root.children}
    maximal_modules_mapping = {}
    ## map each vertex v to a module - where module contains v 
    for child in maximal_modules:
        for v in child:
            maximal_modules_mapping[v] = child
        
    for child in maximal_modules:
        ## from the defintion of a module we only need one vertex from a module 
        ## a module is non empty set  
        v = list(child)[0]
        mutal_adj = set(graph.neighbors(v)) - child
        for i in mutal_adj:
            decomposed_graph.add_edge(child,maximal_modules_mapping[i])
    
    return decomposed_graph



def run_test_on_wiki_graph():
    adj_list = {1:[2,3,4], 2:[1,4,5,6,7], 3:[1,4,5,6,7], 4:[1,2,3,5,6,7],
             5:[2,3,4,6,7], 6:[2,3,4,5,8,9,10,11],
               7:[2,3,4,5,8,9,10,11], 8:[6,7,9,10,11], 9:[6,7,8,10,11],
               10:[6,7,8,9], 11:[6,7,8,9]}
    g = nx.from_dict_of_lists(adj_list)
    md_tree = habib_maurer_algorithm(g)
    p_set = prime_g(md_tree,g)
    preprocessed_graph = create_graph_from_prime_g(md_tree,g)

   
    
def run_test_on_icosahedral_graph():
    g = nx.generators.icosahedral_graph()
    md_tree = habib_maurer_algorithm(g)
    preprocessed_graph = create_graph_from_prime_g(md_tree,g)
  

def quotient_graph(root,g):
    #quotient_g = nx.Graph()
    maximal_modules = {frozenset(get_vertices(child)) for child in root.children}
    vertices = []
    for module in maximal_modules:
        vertices.append(list(module)[0])
    quotient_g = g.subgraph(vertices)
    # nx.draw_networkx(quotient_g)
    # plt.show()
    return quotient_g
    
    

    
def prime_g_helper(root,graph,prime_g_set):
    if root.node_type == NodeType.NORMAL:
        return
    elif root.node_type == NodeType.SERIES or root.node_type == NodeType.PARALLEL:
         for connected_component in root.children:
            prime_g_helper(connected_component,graph,prime_g_set)
    
    elif root.node_type == NodeType.PRIME:
        prime_g_set.append(quotient_graph(root,graph))
        for maximal_module in root.children:
            if len(maximal_module.children) > 1:
                module_vertices = get_vertices(maximal_module)
                g_m = graph.subgraph(module_vertices)
                prime_g_helper(maximal_module,g_m,prime_g_set)

            
def prime_g(root,graph):
    prime_g_set = []
    prime_g_helper(root,graph,prime_g_set)
    return prime_g_set
            

def create_graph_from_prime_g(md_tree,graph):
    prime_g_set = prime_g(md_tree,graph)
    preprocessed_graph = nx.Graph()
    for subgraph in prime_g_set:
        preprocessed_graph = nx.union(preprocessed_graph,subgraph)
    return preprocessed_graph

 
BASE_PATH = Path(__file__).parent


INSTANCES_PATH = (BASE_PATH / "exact-public").resolve() 
RELEVENT_FOR_PREPROCESSING_PATH = (BASE_PATH / "relevant-preprocessing").resolve() 
PREPROCESSED_GRAPH_OUTPUT = (BASE_PATH / "preprocessed_graphs").resolve() 
PREPROCESSING_BEFORE_AFTER_PATH = (BASE_PATH / "preprocessing_before_after").resolve() 

DATASET_FOLDER_NAME = "exact-public"
DATASET_FOLDER_NAME_OUTPUT = "relevant-preprocessing"

def twin_merge(g):
    changed = True
    while changed:
        changed = False
        nodes = sorted(list(g.nodes))
        for n1 in nodes:
            nb1 = set(g.neighbors(n1))
            for n2 in nodes:
                if n1 < n2:
                    nb2 = set(g.neighbors(n2))
                    nbs = nb1 ^ nb2
                    nbs.discard(n1)
                    nbs.discard(n2)
                    if len(nbs) == 0:
                        g.remove_node(n1)
                        changed = True
                        break
    return g

def main():
    is_relevant_preprocessing = True
    if not os.path.exists(RELEVENT_FOR_PREPROCESSING_PATH):
        os.makedirs(RELEVENT_FOR_PREPROCESSING_PATH)
        is_relevant_preprocessing = False
    instances_path = RELEVENT_FOR_PREPROCESSING_PATH if is_relevant_preprocessing else INSTANCES_PATH 
    files = sorted(os.listdir(instances_path))
    if not os.path.exists(PREPROCESSED_GRAPH_OUTPUT):
        os.makedirs(PREPROCESSED_GRAPH_OUTPUT)
    if not os.path.exists(PREPROCESSING_BEFORE_AFTER_PATH):
        os.makedirs(PREPROCESSING_BEFORE_AFTER_PATH)
    for file_name in files:
        if "gr" in file_name:
            input_file_name = (instances_path / file_name).resolve().as_posix()
            output_file_name = (RELEVENT_FOR_PREPROCESSING_PATH / file_name).resolve().as_posix()
            graph = pace_parser.parse(input_file_name)
            graph = twin_merge(graph)
            md_tree = habib_maurer_algorithm(graph)
            if not is_prime(md_tree,graph):
                if not is_relevant_preprocessing:
                    copyfile(input_file_name, output_file_name)
                preprocessed_graph_path = (PREPROCESSED_GRAPH_OUTPUT / f"preprocessed_{file_name}").resolve().as_posix()
                preprocessed_graph = create_graph_from_prime_g(md_tree,graph)
                nx.write_gexf(preprocessed_graph,preprocessed_graph_path)
                file_name_new = f"{str(file_name).split('.')[0]}"
                nx.draw_networkx(graph,with_labels = True)
                before_file_name_path = (PREPROCESSING_BEFORE_AFTER_PATH / f"{file_name_new}_before.png").resolve().as_posix()
                plt.savefig(before_file_name_path)
                plt.clf()
                nx.draw_networkx(preprocessed_graph,with_labels = True)
                after_file_name_path = (PREPROCESSING_BEFORE_AFTER_PATH / f"{file_name_new}_after.png").resolve().as_posix()
                plt.savefig(after_file_name_path)
                plt.clf()
                
            

main()