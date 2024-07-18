import numpy as np
from skimage.metrics import structural_similarity as ssim

from ri_ssim.ri_ssim import (
    _ssim_from_params,
    _ssim_from_params_with_C3,
    range_invariant_structural_similarity,
)
from ri_ssim.ssim import compute_ssim_elements


def test_ssim_with_parameters():
    """Test that `_ssim_from_params_with_C3` returns the same value as skimage SSIM."""
    # prepare arrays
    gt = np.arange(32**2).reshape(32, 32) / (32**2)
    pred = gt + np.random.normal(0, 1, gt.shape)

    # compute ssim
    ssim_skimage = ssim(gt, pred, data_range=gt.max() - gt.min(), gaussian_weights=True)

    # compute ssim elements
    ssim_elements = compute_ssim_elements(
        gt, pred, data_range=gt.max() - gt.min(), gaussian_weights=True
    )

    # compute ssim with C3 = 0
    ssim_elements.C3 = 0
    ssim_with_C3 = _ssim_from_params_with_C3(alpha=1, ssim=ssim_elements)

    assert np.isclose(ssim_skimage, ssim_with_C3, atol=1e-6)


def test_ssim_without_C3():
    """Test that `_ssim_from_params` returns the same value as skimage SSIM."""
    # prepare arrays
    gt = np.arange(32**2).reshape(32, 32) / (32**2)
    pred = gt + np.random.normal(0, 1, gt.shape)

    # compute ssim
    ssim_skimage = ssim(gt, pred, data_range=gt.max() - gt.min(), gaussian_weights=True)

    # compute ssim elements
    ssim_elements = compute_ssim_elements(
        gt, pred, data_range=gt.max() - gt.min(), gaussian_weights=True
    )

    # compute ssim without C3
    ssim_elements.C3 = None
    ssim_without_C3 = _ssim_from_params(alpha=1, ssim=ssim_elements)

    assert np.isclose(ssim_skimage, ssim_without_C3, atol=1e-6)


def test_microssim_same_as_skimage():
    gt = np.arange(10000).reshape(100, 100) / 10000
    pred = gt + np.random.normal(0, 1, gt.shape)
    ssim_skimage = ssim(gt, pred, data_range=gt.max() - gt.min(), gaussian_weights=True)
    ssim_microsim = range_invariant_structural_similarity(
        gt, pred, ri_factor=[1.0], data_range=gt.max() - gt.min(), gaussian_weights=True
    )
    assert np.isclose(ssim_skimage, ssim_microsim, atol=1e-6)
