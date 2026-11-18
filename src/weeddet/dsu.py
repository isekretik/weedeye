"""Система непересекающихся множеств (DSU, union-find), написанная с нуля.

Структура хранится в двух обычных списках Python (parent, rank): никаких
dict / set / готовых контейнеров. Элементы — целые числа 0..n-1.

Две классические оптимизации включаются флагами (нужно для эксперимента,
где сравнивается их вклад):
    compress=True  — сжатие путей (path compression, полная, итеративная);
    by_rank=True   — объединение по рангу (union by rank).

Сложность (m операций find/union над n элементами):
    без оптимизаций           — O(n) на операцию в худшем случае;
    только сжатие путей       — O(log n) амортизированно;
    только ранги              — O(log n) в худшем случае;
    сжатие + ранги            — O(alpha(n)) амортизированно, где alpha —
                                обратная функция Аккермана (<= 4 на практике),
                                то есть m операций стоят почти O(m).
Память: O(n) (два списка по n чисел).
"""


class DSU:
    """Система непересекающихся множеств на элементах 0..n-1."""

    def __init__(self, n: int, compress: bool = True, by_rank: bool = True):
        if n < 0:
            raise ValueError("n должно быть неотрицательным")
        self.n = n
        self.compress = compress
        self.by_rank = by_rank
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # текущее число множеств

    def find(self, x: int) -> int:
        """Корень множества, содержащего x. Итеративно (без рекурсии)."""
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        if self.compress:
            # второй проход: все вершины пути напрямую к корню
            while parent[x] != root:
                nxt = parent[x]
                parent[x] = root
                x = nxt
        return root

    def union(self, a: int, b: int) -> bool:
        """Объединить множества a и b. True, если они были разными."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.by_rank:
            rank = self.rank
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            self.parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
        else:
            self.parent[rb] = ra
        self.count -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        """Лежат ли a и b в одном множестве."""
        return self.find(a) == self.find(b)
