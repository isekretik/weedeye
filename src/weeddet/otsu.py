"""Порог Оцу, написанный с нуля.

Идея: для каждого порога t делим гистограмму на два класса
(уровни <= t и уровни > t) и выбираем t, максимизирующий межклассовую
дисперсию  sigma_b^2(t) = w0 * w1 * (m0 - m1)^2.

Накопленные суммы (w0, sum0) обновляются за O(1) на шаг, поэтому сам
поиск порога стоит O(L), где L — число уровней (256), а не O(L^2).
Построение гистограммы стоит O(N) по числу пикселей.
"""

from typing import List, Sequence

import numpy as np

LEVELS = 256


def histogram_py(values, levels: int = LEVELS) -> List[int]:
    """Гистограмма обычным циклом (эталонная «ручная» реализация), O(N)."""
    hist = [0] * levels
    for v in values.ravel().tolist():
        hist[v] += 1
    return hist


def histogram(values: np.ndarray, levels: int = LEVELS, fast: bool = False) -> List[int]:
    """Гистограмма uint8-массива.

    fast=False (по умолчанию) — собственный цикл Python, без готовых функций.
    fast=True — то же самое через np.bincount; только для ускорения
    экспериментов и сравнения. Тесты проверяют, что результаты совпадают.
    """
    if values.dtype != np.uint8:
        raise ValueError("Ожидается массив uint8 (см. exg.to_uint8)")
    if not fast:
        return histogram_py(values, levels)
    return np.bincount(values.ravel(), minlength=levels).tolist()


def otsu_from_hist(hist: Sequence[int]) -> int:
    """Порог Оцу по гистограмме. Пиксели со значением > t относятся к «переднему плану».

    При нескольких одинаково хороших порогах берётся первый (как в OpenCV).
    Если все пиксели одного уровня (один класс), возвращается этот уровень.
    """
    total = sum(hist)
    if total == 0:
        raise ValueError("Пустая гистограмма")
    sum_all = 0.0
    for i, h in enumerate(hist):
        sum_all += i * h

    w0 = 0
    sum0 = 0.0
    best_var = -1.0
    best_t = 0
    for t, h in enumerate(hist):
        w0 += h
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * h
        m0 = sum0 / w0
        m1 = (sum_all - sum0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best_var:
            best_var = between
            best_t = t
    if best_var < 0:  # единственный непустой уровень
        for t, h in enumerate(hist):
            if h:
                return t
    return best_t


def otsu_threshold(img_u8: np.ndarray, fast_hist: bool = False) -> int:
    """Порог Оцу для uint8-изображения."""
    return otsu_from_hist(histogram(img_u8, fast=fast_hist))


def binarize(img_u8: np.ndarray, threshold: int) -> np.ndarray:
    """Бинарная маска: True там, где значение строго больше порога."""
    return img_u8 > threshold


def segment_vegetation(img_rgb: np.ndarray, fast_hist: bool = False):
    """Пайплайн ExG -> нормировка -> Оцу. Возвращает (маска bool, порог, карта ExG)."""
    from .exg import exg, to_uint8

    index_map = exg(img_rgb)
    u8 = to_uint8(index_map)
    t = otsu_threshold(u8, fast_hist=fast_hist)
    return binarize(u8, t), t, index_map
