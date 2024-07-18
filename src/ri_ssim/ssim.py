"""Calculate the elemnts used for computing the SSIM.

Code adapted from `skimage.metrics.structural_similarity` under BSD-3-Clause license.
See https://github.com/scikit-image/scikit-image.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter
from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.dtype import dtype_range


@dataclass
class SSIM:
    """Dataclass holding the various values necessary for the computation of the
    SSIM."""

    ux: NDArray
    """Weighted mean of the first image."""

    uy: NDArray
    """Weighted mean of the second image."""

    vxy: NDArray
    """Weighted covariance of the two images."""

    vx: NDArray
    """Weighted variance of the first image."""

    vy: NDArray
    """Weighted variance of the second image."""

    C1: float
    """Algorithm parameter, C1."""

    C2: float
    """Algorithm parameter, C2."""

    C3: Optional[float] = None
    """Algorithm parameter, C3."""

    SSIM: Optional[float] = None
    """The SSIM value."""

    luminance: Optional[float] = None
    """The luminance value."""

    contrast: Optional[float] = None
    """The contrast value."""

    structure: Optional[float] = None
    """The structure value."""

    alpha: Optional[float] = None
    """The range-invariant factor."""


def compute_ssim_elements(
    img_x: NDArray,
    img_y: NDArray,
    *,
    win_size: Optional[int] = None,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: bool = False,
    **kwargs: dict[str, float],
) -> SSIM:
    """
    Calculate the elemnts used for computing the SSIM.

    Code adapted from `skimage.metrics.structural_similarity` under BSD-3-Clause
    license.

    Parameters
    ----------
    img_x : numpy.ndarray
        First image being compared.
    img_y : numpy.ndarray
        Second image being compared.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to im2.
    data_range : float, optional
        The data range of the input image (difference between maximum and
        minimum possible values). By default, this is estimated from the image
        data type. This estimate may be wrong for floating-point image data.
        Therefore it is recommended to always pass this scalar value explicitly
        (see note below).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant, see [1]_).
    K2 : float
        Algorithm parameter, K2 (small constant, see [1]_).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.

    Returns
    -------
    SSIMElements dataclass
        The elements used for computing the SSIM (means, variances, covariaces, etc.).
    """
    check_shape_equality(img_x, img_y)
    float_type = _supported_float_type(img_x.dtype)

    if channel_axis is not None:
        raise NotImplementedError(
            "Multichannel images are not supported at this time. "
            "Please set `channel_axis` to None."
        )

    K1 = kwargs.pop("K1", 0.01)  # TODO inherited from skimage, but might make explicit
    K2 = kwargs.pop("K2", 0.03)
    K3 = kwargs.pop("K3", None)

    sigma = kwargs.pop("sigma", 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop("use_sample_covariance", True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7  # backwards compatibility

    if np.any((np.asarray(img_x.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent. "
            "Either ensure that your images are "
            "at least 7x7; or pass win_size explicitly "
            "in the function call, with an odd value "
            "less than or equal to the smaller side of your "
            "images. If your images are multichannel "
            "(with color channels), set channel_axis to "
            "the axis number corresponding to the channels."
        )

    if not (win_size % 2 == 1):
        raise ValueError("Window size must be odd.")

    if data_range is None:
        if np.issubdtype(img_x.dtype, np.floating) or np.issubdtype(
            img_y.dtype, np.floating
        ):
            raise ValueError(
                "Since image dtype is floating point, you must specify "
                "the data_range parameter. Please read the documentation "
                "carefully (including the note). It is recommended that "
                "you always specify the data_range anyway."
            )
        if img_x.dtype != img_y.dtype:
            warn(
                "Inputs have mismatched dtypes. Setting data_range "
                "based on img_x.dtype.",
                stacklevel=2,
            )
        dmin, dmax = dtype_range[img_x.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(img_x.dtype, np.integer) and (img_x.dtype != np.uint8):
            warn(
                "Setting data_range based on img_x.dtype. "
                + f"data_range = {data_range:.0f}. "
                + "Please specify data_range explicitly to avoid mistakes.",
                stacklevel=2,
            )

    ndim = img_x.ndim

    if gaussian_weights:
        filter_func = gaussian
        filter_args = {"sigma": sigma, "truncate": truncate, "mode": "reflect"}
    else:
        filter_func = uniform_filter
        filter_args = {"size": win_size}

    # ndimage filters need floating point data
    img_x = img_x.astype(float_type, copy=False)
    img_y = img_y.astype(float_type, copy=False)

    NP = win_size**ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(img_x, **filter_args)
    uy = filter_func(img_y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(img_x * img_x, **filter_args)
    uyy = filter_func(img_y * img_y, **filter_args)
    uxy = filter_func(img_x * img_y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3 = None if K3 is None else (K3 * R) ** 2

    pad = (win_size - 1) // 2
    ux = ux[pad:-pad, pad:-pad].copy()
    uy = uy[pad:-pad, pad:-pad].copy()
    vxy = vxy[pad:-pad, pad:-pad].copy()
    vx = vx[pad:-pad, pad:-pad].copy()
    vy = vy[pad:-pad, pad:-pad].copy()

    return SSIM(
        ux=ux,
        uy=uy,
        vxy=vxy,
        vx=vx,
        vy=vy,
        C1=C1,
        C2=C2,
        C3=C3,
    )
