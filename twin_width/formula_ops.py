
def neg(x: list):
    return [-e for e in x]


def disj(x: list, y: list):
    return x+y


def conj(x: list, y: list):
    return neg(disj(neg(x), neg(y)))


def xor(x, y):
    return conj([x], [-y])+conj([-x], [y])


def implies(x: list, y: list):
    return disj(neg(x), y)


def equiv(x: list, y: list):
    return disj(conj(x, y), conj(neg(x), neg(y)))
