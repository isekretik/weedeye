import itertools

import numpy as np
import pytest

from weeddet.dsu import DSU

MODES = list(itertools.product([False, True], [False, True]))  # (compress, by_rank)


def test_initial_state():
    d = DSU(5)
    assert d.count == 5
    assert [d.find(i) for i in range(5)] == [0, 1, 2, 3, 4]


def test_empty_and_negative():
    assert DSU(0).count == 0
    with pytest.raises(ValueError):
        DSU(-1)


@pytest.mark.parametrize("compress,by_rank", MODES)
def test_union_and_connected(compress, by_rank):
    d = DSU(6, compress=compress, by_rank=by_rank)
    assert d.union(0, 1) is True
    assert d.union(1, 2) is True
    assert d.union(0, 2) is False  # уже в одном множестве
    assert d.connected(0, 2)
    assert not d.connected(0, 3)
    assert d.count == 4
    d.union(3, 4)
    d.union(2, 4)
    assert d.connected(0, 3)
    assert d.count == 2  # {0,1,2,3,4} и {5}


@pytest.mark.parametrize("compress,by_rank", MODES)
def test_all_modes_give_same_partition(compress, by_rank):
    rng = np.random.default_rng(5)
    n = 200
    pairs = rng.integers(0, n, size=(300, 2))
    ref = DSU(n, compress=True, by_rank=True)
    d = DSU(n, compress=compress, by_rank=by_rank)
    for a, b in pairs:
        ref.union(int(a), int(b))
        d.union(int(a), int(b))
    assert d.count == ref.count
    for i in range(n):
        for j in (0, 17, 99, 150):
            assert d.connected(i, j) == ref.connected(i, j)


def test_path_compression_flattens_chain():
    # без рангов union(i, i+1) строит цепочку; compress её сплющивает
    d = DSU(50, compress=True, by_rank=False)
    for i in range(49):
        d.union(i + 1, i)  # корень каждый раз ставится под новым
    d.find(0)
    # после find все вершины пути указывают прямо на корень
    root = d.find(0)
    assert all(d.parent[i] == root or i == root for i in range(50))


def test_deep_chain_is_not_recursive():
    # без оптимизаций цепочка длиной 100000: рекурсивный find упал бы
    n = 100_000
    d = DSU(n, compress=False, by_rank=False)
    for i in range(n - 1):
        d.union(i + 1, i)
    assert d.find(0) == d.find(n - 1)
    assert d.count == 1


def test_rank_keeps_trees_shallow():
    d = DSU(1024, compress=False, by_rank=True)
    for i in range(1023):
        d.union(i, i + 1)

    def depth(x):
        k = 0
        while d.parent[x] != x:
            x = d.parent[x]
            k += 1
        return k

    assert max(depth(i) for i in range(1024)) <= 10  # log2(1024)
