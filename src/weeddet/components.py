"""Компоненты связности бинарной маски через систему непересекающихся множеств.

Алгоритм (label_dsu):
  1. Каждому пикселю маски соответствует элемент DSU с номером r * W + c.
  2. Проход по маске в строчном порядке: для каждого «живого» пикселя
     объединяем его с уже просмотренными соседями — слева и сверху
     (4-связность) плюс верхними диагоналями (8-связность).
  3. Второй проход: корень каждого пикселя даёт номер компоненты; метки
     1..k выдаются в порядке первого появления корня при строчном обходе
     (как в scipy.ndimage.label).

Сложность: O(N * alpha(N)) по времени, где N = H * W — число пикселей
(на практике почти линейно); память O(N) (parent, rank, метки).

Для сравнения есть вариант на поиске в ширину (label_bfs). Он не использует
готовых очередей: стек и «очередь» — обычный список с индексом. O(N), O(N).
"""

from typing import Tuple

import numpy as np

from .dsu import DSU

# смещения уже просмотренных соседей (строка, столбец)
_NEIGHBORS_PAST_4 = ((0, -1), (-1, 0))
_NEIGHBORS_PAST_8 = ((0, -1), (-1, -1), (-1, 0), (-1, 1))

# все соседи — для BFS
_NEIGHBORS_ALL_4 = ((0, -1), (0, 1), (-1, 0), (1, 0))
_NEIGHBORS_ALL_8 = _NEIGHBORS_ALL_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _check(mask: np.ndarray, connectivity: int):
    if mask.ndim != 2:
        raise ValueError("Ожидается двумерная маска (H, W)")
    if connectivity not in (4, 8):
        raise ValueError("connectivity должна быть 4 или 8")


def label_dsu(
    mask: np.ndarray,
    connectivity: int = 8,
    compress: bool = True,
    by_rank: bool = True,
) -> Tuple[np.ndarray, int]:
    """Разметить компоненты связности через DSU.

    Возвращает (labels, n): labels — int32 (H, W), 0 = фон, 1..n = номер
    компоненты; n — число компонент.
    """
    _check(mask, connectivity)
    h, w = mask.shape
    nb = _NEIGHBORS_PAST_8 if connectivity == 8 else _NEIGHBORS_PAST_4
    fg = mask.astype(bool).tolist()  # список списков: быстрее, чем индексация numpy в цикле

    dsu = DSU(h * w, compress=compress, by_rank=by_rank)
    union = dsu.union

    for r in range(h):
        row = fg[r]
        for c in range(w):
            if not row[c]:
                continue
            i = r * w + c
            for dr, dc in nb:
                rr, cc = r + dr, c + dc
                if rr < 0 or cc < 0 or cc >= w:
                    continue
                if fg[rr][cc]:
                    union(i, rr * w + cc)

    # второй проход: корни -> метки 1..n в порядке первого появления
    labels = np.zeros((h, w), dtype=np.int32)
    root_label = [0] * (h * w)  # root_label[корень] = метка (0 = ещё нет)
    n = 0
    find = dsu.find
    for r in range(h):
        row = fg[r]
        for c in range(w):
            if not row[c]:
                continue
            root = find(r * w + c)
            lab = root_label[root]
            if lab == 0:
                n += 1
                lab = n
                root_label[root] = lab
            labels[r, c] = lab
    return labels, n


def label_bfs(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Разметка компонент поиском в ширину (эталон для сравнения с DSU)."""
    _check(mask, connectivity)
    h, w = mask.shape
    nb = _NEIGHBORS_ALL_8 if connectivity == 8 else _NEIGHBORS_ALL_4
    fg = mask.astype(bool).tolist()
    lab = [[0] * w for _ in range(h)]
    n = 0
    for r0 in range(h):
        for c0 in range(w):
            if not fg[r0][c0] or lab[r0][c0]:
                continue
            n += 1
            lab[r0][c0] = n
            queue = [(r0, c0)]
            head = 0  # «очередь» на списке: индекс головы вместо pop(0)
            while head < len(queue):
                r, c = queue[head]
                head += 1
                for dr, dc in nb:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and fg[rr][cc] and not lab[rr][cc]:
                        lab[rr][cc] = n
                        queue.append((rr, cc))
    return np.array(lab, dtype=np.int32), n
