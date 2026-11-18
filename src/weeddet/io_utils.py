"""Ввод-вывод изображений (здесь допустимы готовые библиотеки)."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path) -> np.ndarray:
    """Загрузить изображение как массив uint8 формы (H, W, 3), порядок каналов RGB."""
    with Image.open(Path(path)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def load_mask(path) -> np.ndarray:
    """Загрузить бинарную маску как bool-массив (H, W): любой ненулевой пиксель = True."""
    with Image.open(Path(path)) as im:
        arr = np.asarray(im.convert("L"))
    return arr > 0


def load_cwfid_vegetation_mask(path) -> np.ndarray:
    """Маска растительности CWFID как bool-массив: True = растение.

    ВАЖНО: в файлах масок CWFID значения инвертированы относительно ожидаемых:
    True (белый) занимает 82-92% кадра, то есть это почва, а растения = False.
    Полярность установлена измерением на всех 60 снимках (README датасета её
    не описывает), поэтому здесь маска инвертируется.
    """
    return ~load_mask(path)


def save_rgb(path, img: np.ndarray) -> None:
    """Сохранить массив uint8 (H, W, 3) или (H, W) в файл."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(Path(path))
