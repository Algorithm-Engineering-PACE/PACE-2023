def disj(x, y):
    # Simply join the clauses, since a clause is a disjunction of literals
    return x + y

def conj(x: list, y: list):
    # A conjunction of clauses is a list of those clauses
    if isinstance(x, list) and isinstance(y, list):
        return [x, y]

def implies(x, y):
    # (x => y) is equivalent to (~x v y)
    if isinstance(x, int):
        if isinstance(y, int):
            return [[-x, y]]
        else: # y is a (or list)
            return [[-x] + y]

    elif len(x) == 2 and len(y) == 1:
        return [disj([-x[0][0]], [-x[1][0]]) + y]
    else:
        assert False

def equiv(x: int, y):
    """
    :x list contains one var
    :y list of list[Int]
    (x <=> y) is equivalent to (x => y) ^ (y => x)
    """
    if isinstance(x, int) and isinstance(y[0], int):
        return [[-x, *y], *[[-z] for z in y]]
    elif isinstance(x, int) and isinstance(y[0], list):
        assert False
    else:
        assert False
