import numpy as np
import pytest

from weeddet.baselines import otsu_cv2, otsu_skimage
from weeddet.exg import exg, to_uint8
from weeddet.otsu import binarize, histogram, otsu_from_hist, otsu_threshold
from weeddet.synthetic import make_scene


def test_histogram_fast_equals_loop():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(30, 40), dtype=np.uint8)
    assert histogram(a, fast=True) == histogram(a, fast=False)
    assert sum(histogram(a)) == a.size


def test_bimodal_threshold_between_modes():
    hist = [0] * 256
    hist[40] = 500
    hist[200] = 500
    t = otsu_from_hist(hist)
    assert 40 <= t < 200


def test_single_level_image():
    hist = [0] * 256
    hist[77] = 100
    assert otsu_from_hist(hist) == 77


def test_empty_histogram_raises():
    with pytest.raises(ValueError):
        otsu_from_hist([0] * 256)


def test_binarize_is_strictly_greater():
    a = np.array([[10, 11, 12]], dtype=np.uint8)
    assert binarize(a, 11).tolist() == [[False, False, True]]


@pytest.mark.parametrize("seed", range(5))
def test_matches_opencv_and_skimage_on_random(seed):
    rng = np.random.default_rng(seed)
    # смесь двух нормальных распределений
    a = np.concatenate([rng.normal(70, 15, 4000), rng.normal(180, 20, 2500)])
    img = np.clip(a, 0, 255).astype(np.uint8).reshape(50, 130)
    mine = otsu_threshold(img)
    assert abs(mine - otsu_cv2(img)) <= 1
    assert abs(mine - otsu_skimage(img)) <= 1


@pytest.mark.parametrize("seed", range(5))
def test_matches_opencv_on_synthetic_exg(seed):
    scene = make_scene(seed=seed)
    u8 = to_uint8(exg(scene.image))
    mine = otsu_threshold(u8)
    assert abs(mine - otsu_cv2(u8)) <= 1
    assert abs(mine - otsu_skimage(u8)) <= 1
