import time

from networkx import Graph
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from threading import Timer
import tools

from twin_width.encoding2 import TwinWidthEncoding2

class EncodingEvaluator(TwinWidthEncoding2):
    def __init__(self, card_enc: EncType):
        super().__init__(card_enc)

    def encode_order(self, n):
        self.formula.append([self.ord[n][n]])
        # Assign one node to each time step
        for i in range(1, n):
            self.formula.extend(CardEnc.equals([self.ord[i][j] for j in range(1, n+1)], vpool=self.pool, encoding=self.card_enc))

        # Make sure each node is assigned only once...
        for i in range(1, n+1):
            self.formula.extend(CardEnc.atmost([self.ord[j][i] for j in range(1, n + 1)], vpool=self.pool, encoding=self.card_enc))

    def encode_merge(self, n, d):
        # Exclude root
        for i in range(1, n-d + 1):
            self.formula.extend(CardEnc.equals([self.merge[i][j] for j in range(i + 1, n + 1)],
                                                vpool=self.pool, encoding=self.card_enc))
        merge_prime = [{} for _ in range(0, n+1)]
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                merge_prime[i][j] = self.pool.id(f"merge_prime{i}_{j}")

        for i in range(1, n+1):
            for j in range(1, n+1):
                for k in range(1, n+1):
                    if j == k:
                        continue
                    if k > j:
                        self.formula.append([-self.ord[i][j], merge_prime[i][k]])
                    else:
                        self.formula.append([-self.ord[i][j], -merge_prime[i][k]])


                # nb_vars = [self.ord[i][x] for x in nb]
                # self.formula.append([-merge_prime[i][j], *nb_vars])

        for i in range(1, n+1):
            for j in range(1, n+1):
                for k in range(i+1, n-d+1):
                    self.formula.append([-self.ord[k][j], -merge_prime[i][j], self.merge_ord[i][k]])
                    self.formula.append([-self.ord[k][j], merge_prime[i][j], -self.merge_ord[i][k]])

        for i in range(1, n+1):
            for j in range(i+1, n-d+1):
                    self.formula.append([-self.merge[i][j], self.merge_ord[i][j]])
