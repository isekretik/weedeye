import numpy as np
from PIL import Image

from weeddet.io_utils import load_cwfid_vegetation_mask, load_mask, load_rgb, save_rgb


def test_load_mask_nonzero_is_true(tmp_path):
    arr = np.zeros((4, 6), dtype=np.uint8)
    arr[1:3, 2:5] = 255
    p = tmp_path / "m.png"
    Image.fromarray(arr).save(p)
    m = load_mask(p)
    assert m.dtype == bool and np.array_equal(m, arr > 0)


def test_cwfid_mask_is_inverted(tmp_path):
    # В CWFID True = почва, растение = False; загрузчик обязан инвертировать.
    arr = np.full((4, 6), 255, dtype=np.uint8)
    arr[1:3, 2:5] = 0  # «растение»
    p = tmp_path / "m.png"
    Image.fromarray(arr).save(p)
    veg = load_cwfid_vegetation_mask(p)
    assert veg[1:3, 2:5].all()
    assert veg.sum() == 6


def test_rgb_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    p = tmp_path / "sub" / "a.png"
    save_rgb(p, img)
    assert np.array_equal(load_rgb(p), img)
