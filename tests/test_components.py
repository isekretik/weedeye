import numpy as np
import pytest

from weeddet.baselines import label_cv2, label_scipy
from weeddet.components import label_bfs, label_dsu
from weeddet.otsu import segment_vegetation
from weeddet.synthetic import make_scene


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Два массива меток задают одно и то же разбиение (метки могут различаться)."""
    if a.shape != b.shape or not np.array_equal(a > 0, b > 0):
        return False
    fg = a > 0
    pairs = np.unique(np.stack([a[fg], b[fg]], axis=1), axis=0)
    # взаимно однозначное соответствие меток
    return len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1]))


def test_empty_mask():
    m = np.zeros((5, 6), dtype=bool)
    lab, n = label_dsu(m)
    assert n == 0 and lab.sum() == 0


def test_full_mask_is_one_component():
    m = np.ones((7, 9), dtype=bool)
    lab, n = label_dsu(m, 4)
    assert n == 1 and (lab == 1).all()


def test_single_pixel_and_edges():
    m = np.zeros((4, 4), dtype=bool)
    m[0, 0] = m[3, 3] = True
    _, n = label_dsu(m, 8)
    assert n == 2


def test_diagonal_4_vs_8():
    m = np.eye(5, dtype=bool)
    assert label_dsu(m, 4)[1] == 5
    assert label_dsu(m, 8)[1] == 1


def test_known_small_example():
    m = np.array(
        [
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
        ],
        dtype=bool,
    )
    lab, n = label_dsu(m, 4)
    assert n == 4
    # метки идут в порядке первого появления при строчном обходе
    assert lab[0, 0] == lab[0, 1] == lab[1, 1] == 1
    assert lab[0, 4] == lab[1, 4] == 2
    assert lab[2, 3] == lab[3, 3] == lab[3, 4] == 3
    assert lab[3, 0] == 4


def test_bad_arguments():
    with pytest.raises(ValueError):
        label_dsu(np.zeros((3, 3, 3), dtype=bool))
    with pytest.raises(ValueError):
        label_dsu(np.zeros((3, 3), dtype=bool), connectivity=6)
    with pytest.raises(ValueError):
        label_bfs(np.zeros((3, 3), dtype=bool), connectivity=5)


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("seed", range(6))
def test_random_masks_match_all_references(seed, connectivity):
    rng = np.random.default_rng(seed)
    density = [0.2, 0.4, 0.55][seed % 3]
    m = rng.random((40, 55)) < density
    mine, n = label_dsu(m, connectivity)
    for ref_lab, ref_n in (
        label_scipy(m, connectivity),
        label_cv2(m, connectivity),
        label_bfs(m, connectivity),
    ):
        assert n == ref_n
        assert same_partition(mine, ref_lab)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_dsu_labels_equal_scipy_exactly(connectivity):
    # порядок меток тоже совпадает: оба нумеруют по первому появлению
    m = np.random.default_rng(11).random((30, 30)) < 0.45
    mine, _ = label_dsu(m, connectivity)
    ref, _ = label_scipy(m, connectivity)
    assert np.array_equal(mine, ref)


@pytest.mark.parametrize("compress,by_rank", [(False, False), (True, False), (False, True), (True, True)])
def test_optimizations_do_not_change_result(compress, by_rank):
    m = np.random.default_rng(3).random((35, 35)) < 0.5
    base, n0 = label_dsu(m, 8, compress=True, by_rank=True)
    lab, n = label_dsu(m, 8, compress=compress, by_rank=by_rank)
    assert n == n0 and np.array_equal(lab, base)


@pytest.mark.parametrize("seed", range(5))
def test_counts_synthetic_plants(seed):
    s = make_scene(seed=seed)
    lab, n = label_dsu(s.mask, 8)
    assert n == s.n_objects  # эллипсы разделены зазором


@pytest.mark.parametrize("seed", range(3))
def test_pipeline_exg_otsu_dsu_counts_plants(seed):
    s = make_scene(seed=seed)
    mask, _, _ = segment_vegetation(s.image)
    _, n = label_dsu(mask, 8)
    # шум может дать мелкие лишние компоненты, но порядок величины верный
    assert s.n_objects <= n <= s.n_objects + 25
