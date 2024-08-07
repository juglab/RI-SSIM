"""
Structural similarity index code adapted from skimage.metrics.structural_similarity
"""

import numpy as np
from scipy.ndimage import uniform_filter

from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.dtype import dtype_range


def structural_similarity_dict(
    img_x,
    img_y,
    *,
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    return_contrast_sensitivity=False,
    **kwargs,
):
    """
    Compute the mean structural similarity index between two images.
    Please pay attention to the `data_range` parameter with floating-point images.

    Parameters
    ----------
    img_x, img_y : ndarray
        Images. Any dimensionality with same shape.
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

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, also return the full structural similarity image.

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
    mssim : float
        The mean structural similarity index over the image.
    grad : ndarray
        The gradient of the structural similarity between im1 and im2 [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    If `data_range` is not specified, the range is automatically guessed
    based on the image data type. However for floating-point image data, this
    estimate yields a result double the value of the desired range, as the
    `dtype_range` in `skimage.util.dtype.py` has defined intervals from -1 to
    +1. This yields an estimate of 2, instead of 1, which is most often
    required when working with image data (as negative light intensities are
    nonsensical). In case of working with YCbCr-like color data, note that
    these ranges are different per channel (Cb and Cr have double the range
    of Y), so one cannot calculate a channel-averaged SSIM with a single call
    to this function, as identical ranges are assumed for each channel.

    To match the implementation of Wang et al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, `use_sample_covariance` to False, and
    specify the `data_range` argument.

    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_ssim`` to
        ``skimage.metrics.structural_similarity``.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arxiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`

    """
    check_shape_equality(img_x, img_y)
    float_type = _supported_float_type(img_x.dtype)

    if channel_axis is not None:
        raise NotImplementedError(
            "Multichannel images are not supported at this time. "
            "Please set `channel_axis` to None."
        )

    K1 = kwargs.pop("K1", 0.01)
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
                "Inputs have mismatched dtypes. Setting data_range based on img_x.dtype.",
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
    return {
        "ux": ux,
        "uy": uy,
        "vxy": vxy,
        "vx": vx,
        "vy": vy,
        "C1": C1,
        "C2": C2,
        "C3": C3,
    }

    # A1, A2, B1, B2 = (
    #     2 * ux * uy + C1,
    #     2 * vxy + C2,
    #     ux**2 + uy**2 + C1,
    #     vx + vy + C2,
    # )

    # D = B1 * B2
    # S = (A1 * A2) / D

    # # to avoid edge effects will ignore filter radius strip around edges
    # pad = (win_size - 1) // 2

    # # compute (weighted) mean of ssim. Use float64 for accuracy.
    # mssim = crop(S, pad).mean(dtype=np.float64)

    # if gradient:
    #     # The following is Eqs. 7-8 of Avanaki 2009.
    #     grad = filter_func(A1 / D, **filter_args) * im1
    #     grad += filter_func(-S / B2, **filter_args) * img_y
    #     grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D, **filter_args)
    #     grad *= 2 / im1.size

    #     if full:
    #         return mssim, grad, S
    #     else:
    #         return mssim, grad
    # else:
    #     if full:
    #         return mssim, S
    #     else:
    #         return mssim
