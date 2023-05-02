from enum import Enum
import networkx as nx


## sagemath implementation for graph modular decomposition 

class NodeType(Enum):
    """
    NodeType is an enumeration class used to define the various types of nodes
    in modular decomposition tree.

    The various node types defined are

    - ``PARALLEL`` -- indicates the node is a parallel module (G is not connected)

    - ``SERIES`` -- indicates the node is a series module (G complement is not connected)

    - ``PRIME`` -- indicates the node is a prime module (G and G complement is connected)

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


## my functions


def get_vertices(component_root):
    
    vertices = []

    def recurse_component(node, vertices):
        if node.node_type == NodeType.NORMAL:
            vertices.append(node.children[0])
            return
        for child in node.children:
            recurse_component(child, vertices)

    recurse_component(component_root, vertices)
    return vertices


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



def is_prime(md_tree,graph):
    return md_tree.node_type == NodeType.PRIME and len(md_tree.children) == graph.number_of_nodes()



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
    p_set = prime_g(md_tree,g)
    preprocessed_graph = create_graph_from_prime_g(md_tree,g)
  

def quotient_graph(root,g):
    maximal_modules = {frozenset(get_vertices(child)) for child in root.children}
    vertices = []
    for module in maximal_modules:
        vertices.append(list(module)[0])
    quotient_g = g.subgraph(vertices)
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

 # this function is used before calculating tww 

def preproccess(graph):
    output_graph = graph 
    md_tree = habib_maurer_algorithm(graph)
    if not is_prime(md_tree,graph):
        output_graph = create_graph_from_prime_g(md_tree,graph)  
    return output_graph
