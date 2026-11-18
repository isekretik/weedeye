"""Эксперимент 0: сегментация растительности (ExG + Оцу) на CWFID.

Для каждого из 60 снимков сравнивается маска «ExG + Оцу» с маской
растительности датасета: IoU, F1 по пикселям, расхождение порога с OpenCV
и время работы. Результат пишется в results/tables/exp0_cwfid.csv.

Запуск из корня репозитория:
    python experiments/exp0_segmentation_cwfid.py [путь/к/cwfid]
По умолчанию: data/raw/cwfid. Случайности нет, seed не нужен: метод детерминирован.
"""

import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from weeddet.baselines import otsu_cv2  # noqa: E402
from weeddet.exg import exg, to_uint8  # noqa: E402
from weeddet.io_utils import load_cwfid_vegetation_mask, load_rgb  # noqa: E402
from weeddet.otsu import binarize, otsu_threshold  # noqa: E402


def main(cwfid_dir: Path) -> None:
    images = sorted((cwfid_dir / "images").glob("*_image.png"))
    if not images:
        sys.exit(f"Не найдены снимки в {cwfid_dir}/images. См. data/README.md")

    rows = []
    for img_path in images:
        stem = img_path.name.split("_")[0]
        img = load_rgb(img_path)
        gt = load_cwfid_vegetation_mask(cwfid_dir / "masks" / f"{stem}_mask.png")

        t0 = time.perf_counter()
        u8 = to_uint8(exg(img))
        t = otsu_threshold(u8)  # собственная гистограмма (без np.bincount)
        pred = binarize(u8, t)
        dt = time.perf_counter() - t0

        inter = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        p = inter / max(int(pred.sum()), 1)
        r = inter / max(int(gt.sum()), 1)
        rows.append(
            {
                "image": stem,
                "threshold": t,
                "threshold_cv2": otsu_cv2(u8),
                "iou": inter / union,
                "f1": 2 * p * r / max(p + r, 1e-12),
                "time_s": dt,
            }
        )

    out = ROOT / "results" / "tables" / "exp0_cwfid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    iou = np.array([r["iou"] for r in rows])
    f1 = np.array([r["f1"] for r in rows])
    dthr = max(abs(r["threshold"] - r["threshold_cv2"]) for r in rows)
    tm = np.array([r["time_s"] for r in rows])
    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print(f"Снимков: {len(rows)}")
    print(f"IoU  среднее/мин/макс: {iou.mean():.3f} / {iou.min():.3f} / {iou.max():.3f}")
    print(f"F1   среднее: {f1.mean():.3f}")
    print(f"Макс. расхождение порога с OpenCV: {dthr}")
    print(f"Время на снимок, с (среднее): {tm.mean():.3f}")
    print(f"Таблица: {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "cwfid")
