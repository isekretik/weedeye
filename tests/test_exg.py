import numpy as np
import pytest

from weeddet.exg import exg, to_uint8


def test_gray_pixels_have_zero_exg():
    img = np.full((4, 5, 3), 120, dtype=np.uint8)
    assert np.allclose(exg(img), 0.0, atol=1e-6)


def test_green_higher_than_soil():
    green = np.array([[[50, 150, 40]]], dtype=np.uint8)
    soil = np.array([[[125, 90, 65]]], dtype=np.uint8)
    assert exg(green)[0, 0] > 0.3
    assert exg(soil)[0, 0] < 0.0
    assert exg(green)[0, 0] > exg(soil)[0, 0]


def test_known_value():
    # R=0, G=100, B=0 -> r=0, g=1, b=0 -> ExG = 2
    img = np.array([[[0, 100, 0]]], dtype=np.uint8)
    assert exg(img)[0, 0] == pytest.approx(2.0, abs=1e-6)


def test_brightness_invariance():
    img = np.array([[[40, 100, 30]]], dtype=np.uint8)
    brighter = (img * 2).astype(np.uint8)
    assert exg(img)[0, 0] == pytest.approx(exg(brighter)[0, 0], abs=1e-6)


def test_black_pixel_is_finite():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    assert np.all(np.isfinite(exg(img)))


def test_bad_shape():
    with pytest.raises(ValueError):
        exg(np.zeros((4, 4), dtype=np.uint8))


def test_to_uint8_range_and_const():
    m = np.linspace(-1, 2, 50).reshape(5, 10)
    u = to_uint8(m)
    assert u.dtype == np.uint8 and u.min() == 0 and u.max() == 255
    assert np.all(to_uint8(np.full((3, 3), 0.7)) == 0)
