"""Индекс избыточной зелени ExG (Excess Green).

ExG = 2g - r - b, где r, g, b — хроматические координаты:
    r = R / (R + G + B),  g = G / (R + G + B),  b = B / (R + G + B).

Нормировка на сумму каналов делает индекс устойчивее к общей яркости
освещения. Сложность: O(N) по числу пикселей, память O(N).
"""

import numpy as np

EPS = 1e-9


def exg(img_rgb: np.ndarray) -> np.ndarray:
    """Вычислить ExG для RGB-изображения.

    Параметры
    ---------
    img_rgb : массив (H, W, 3), uint8 или float, порядок каналов RGB.

    Возвращает
    ----------
    float64-массив (H, W). Диапазон теоретически [-1, 2]; для чисто
    серых пикселей (R = G = B) значение равно 0, для зелёных положительно.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Ожидается массив формы (H, W, 3)")
    a = img_rgb.astype(np.float64)
    s = a.sum(axis=2) + EPS
    r = a[..., 0] / s
    g = a[..., 1] / s
    b = a[..., 2] / s
    return 2.0 * g - r - b


def to_uint8(index_map: np.ndarray) -> np.ndarray:
    """Линейно растянуть карту индекса на диапазон 0..255 (uint8).

    Нужно, чтобы применить порог Оцу, который работает с гистограммой
    из 256 уровней. Min-max нормировка выполняется по каждому изображению.
    Для константной карты возвращается массив нулей.
    """
    lo = float(index_map.min())
    hi = float(index_map.max())
    if hi - lo < EPS:
        return np.zeros(index_map.shape, dtype=np.uint8)
    scaled = (index_map - lo) / (hi - lo) * 255.0
    return np.rint(scaled).astype(np.uint8)
