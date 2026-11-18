"""Генератор синтетических сцен «почва + растения» с точной разметкой.

Нужен, чтобы проверять алгоритмы на данных с заведомо известным ответом:
число объектов, их маски и положение известны точно. Объекты (эллипсы)
не пересекаются и разделены зазором, поэтому истинное число компонент
связности равно числу растений. Результат полностью определяется seed.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Scene:
    image: np.ndarray            # (H, W, 3) uint8, RGB
    mask: np.ndarray             # (H, W) bool, True = растение
    labels: np.ndarray           # (H, W) int32, 0 = фон, 1..n = номер растения
    centers: List[Tuple[float, float]]  # (row, col) центров эллипсов
    n_objects: int


def _smooth_noise(rng, h, w, cell, amp):
    """Низкочастотный шум: случайная сетка, растянутая до размера кадра."""
    gh, gw = h // cell + 2, w // cell + 2
    grid = rng.normal(0.0, amp, size=(gh, gw))
    big = np.kron(grid, np.ones((cell, cell)))
    return big[:h, :w]


def make_scene(
    h: int = 256,
    w: int = 384,
    n_plants: int = 12,
    seed: int = 0,
    radius_range: Tuple[int, int] = (6, 20),
    min_gap: int = 4,
    noise_sigma: float = 8.0,
) -> Scene:
    """Создать одну сцену. Если места не хватает, растений может получиться меньше n_plants
    (фактическое число лежит в Scene.n_objects)."""
    rng = np.random.default_rng(seed)

    # --- почва: коричневый фон, плавное освещение, мелкий шум ---
    soil_base = np.array([125.0, 90.0, 65.0])
    img = np.empty((h, w, 3), dtype=np.float64)
    img[:] = soil_base
    light = 1.0 + _smooth_noise(rng, h, w, cell=32, amp=0.06)
    gradient = np.linspace(0.92, 1.08, w)[None, :]
    img *= (light * gradient)[..., None]

    # --- размещение эллипсов без пересечений (выборка с отклонением) ---
    placed = []  # (cy, cx, a, b, theta)
    attempts = 0
    r_min, r_max = radius_range
    while len(placed) < n_plants and attempts < 5000:
        attempts += 1
        a = rng.uniform(r_min, r_max)
        b = rng.uniform(r_min, a)
        margin = int(np.ceil(a)) + 2
        if h <= 2 * margin or w <= 2 * margin:
            continue
        cy = rng.uniform(margin, h - margin)
        cx = rng.uniform(margin, w - margin)
        theta = rng.uniform(0, np.pi)
        ok = True
        for (py, px, pa, pb, _) in placed:
            if np.hypot(cy - py, cx - px) <= a + pa + min_gap:
                ok = False
                break
        if ok:
            placed.append((cy, cx, a, b, theta))

    mask = np.zeros((h, w), dtype=bool)
    labels = np.zeros((h, w), dtype=np.int32)
    centers = []

    # --- растения: зелёный цвет с индивидуальным оттенком и затенением к краям ---
    for idx, (cy, cx, a, b, theta) in enumerate(placed, start=1):
        y0, y1 = int(max(cy - a - 1, 0)), int(min(cy + a + 2, h))
        x0, x1 = int(max(cx - a - 1, 0)), int(min(cx + a + 2, w))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dy, dx = yy - cy, xx - cx
        u = (dx * np.cos(theta) + dy * np.sin(theta)) / a
        v = (-dx * np.sin(theta) + dy * np.cos(theta)) / b
        rho2 = u * u + v * v
        inside = rho2 <= 1.0
        color = np.array([55.0, 140.0, 45.0]) + rng.uniform(-15, 15, size=3)
        shade = 1.0 - 0.3 * np.sqrt(np.clip(rho2, 0, 1))
        patch = img[y0:y1, x0:x1]
        patch[inside] = (color[None, :] * shade[inside][:, None])
        mask[y0:y1, x0:x1] |= inside
        labels[y0:y1, x0:x1][inside] = idx
        centers.append((float(cy), float(cx)))

    # --- шум сенсора ---
    img += rng.normal(0.0, noise_sigma, size=img.shape)
    image = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    return Scene(image=image, mask=mask, labels=labels, centers=centers, n_objects=len(placed))


def make_dataset(n: int, seed: int = 0, **kwargs) -> List[Scene]:
    """Набор из n сцен; i-я сцена использует seed + i."""
    return [make_scene(seed=seed + i, **kwargs) for i in range(n)]
