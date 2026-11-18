"""Готовые реализации для СРАВНЕНИЯ. Ядро проекта их не использует."""

import numpy as np


def otsu_cv2(img_u8: np.ndarray) -> int:
    """Порог Оцу из OpenCV (маска: значение > порога)."""
    import cv2

    t, _ = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def otsu_skimage(img_u8: np.ndarray) -> int:
    """Порог Оцу из scikit-image (маска: значение > порога)."""
    from skimage.filters import threshold_otsu

    return int(threshold_otsu(img_u8))


def label_scipy(mask: np.ndarray, connectivity: int = 8):
    """Компоненты связности из SciPy. Возвращает (labels, n)."""
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=int) if connectivity == 8 else None
    labels, n = ndimage.label(mask, structure=structure)
    return labels.astype(np.int32), int(n)


def label_cv2(mask: np.ndarray, connectivity: int = 8):
    """Компоненты связности из OpenCV. Возвращает (labels, n) без учёта фона."""
    import cv2

    n_all, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    return labels.astype(np.int32), int(n_all) - 1
