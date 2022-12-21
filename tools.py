import networkx as nx


def prime_paley(p):
    G = nx.Graph()

    square_set = {(x ** 2) % p for x in range(1, p)}

    for x in range(p):
        for y in range(x+1, p):
            if y - x in square_set:
                G.add_edge(x, y)

    return G
