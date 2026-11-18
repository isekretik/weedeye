import numpy as np
import pytest

from weeddet.otsu import segment_vegetation
from weeddet.synthetic import make_scene


def test_reproducible_by_seed():
    a, b = make_scene(seed=42), make_scene(seed=42)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.labels, b.labels)
    assert not np.array_equal(a.image, make_scene(seed=43).image)


def test_shapes_and_dtypes():
    s = make_scene(h=100, w=150, n_plants=5, seed=3)
    assert s.image.shape == (100, 150, 3) and s.image.dtype == np.uint8
    assert s.mask.shape == (100, 150) and s.mask.dtype == bool
    assert s.labels.shape == (100, 150)


def test_labels_consistent_with_mask_and_count():
    s = make_scene(seed=7, n_plants=10)
    assert np.array_equal(s.labels > 0, s.mask)
    assert len(np.unique(s.labels)) - 1 == s.n_objects
    assert s.n_objects == len(s.centers)
    assert 1 <= s.n_objects <= 10


@pytest.mark.parametrize("seed", range(5))
def test_exg_otsu_segments_synthetic_well(seed):
    s = make_scene(seed=seed)
    pred, _, _ = segment_vegetation(s.image)
    inter = np.logical_and(pred, s.mask).sum()
    union = np.logical_or(pred, s.mask).sum()
    assert inter / union > 0.9
