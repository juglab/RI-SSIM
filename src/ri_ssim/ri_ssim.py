"""Range invariant structural similarity index."""

from copy import copy
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ._mse_ri_factor import get_mse_based_factor
from .ssim import SSIM, compute_ssim_elements


def _ssim_from_params_with_C3(
    alpha, ssim: SSIM, return_individual_components=False
) -> Union[float, SSIM]:
    """Compute SSIM with C3.

    `ssim` is obtained from calling `compute_ssim_elements`.

    Parameters
    ----------
    alpha : float
        The range-invariant factor.
    ssim : SSIM
        The SSIM elements.
    return_individual_components : bool, optional
        If True, return the individual components of the SSIM value.

    Returns
    -------
    Union[float, SSIM]
        The SSIM value or the individual components of the SSIM.
    """
    lum_num = 2 * alpha * ssim.ux * ssim.uy + ssim.C1
    lum_denom = ssim.ux**2 + (alpha**2) * ssim.uy**2 + ssim.C1

    contrast_num = 2 * alpha * np.sqrt(ssim.vx * ssim.vy) + ssim.C2
    contrast_denom = ssim.vx + (alpha**2) * ssim.vy + ssim.C2

    structure_denom = alpha * np.sqrt(ssim.vx * ssim.vy) + ssim.C3
    structure_num = alpha * ssim.vxy + ssim.C3

    num = lum_num * contrast_num * structure_num
    denom = lum_denom * contrast_denom * structure_denom
    S = num / denom

    if return_individual_components:
        return SSIM(
            SSIM=S,
            luminance=lum_num / lum_denom,
            contrast=contrast_num / contrast_denom,
            structure=structure_num / structure_denom,
            alpha=alpha,
            ux=ssim.ux,
            uy=ssim.uy,
            vx=ssim.vx,
            vy=ssim.vy,
            vxy=ssim.vxy,
            C1=ssim.C1,
            C2=ssim.C2,
            C3=ssim.C3,
        )

    return np.mean(S)


def _ssim_from_params(
    alpha, ssim: SSIM, return_individual_components=False
) -> Union[float, SSIM]:
    """Compute SSIM without C3.

    `ssim` is obtained from calling `compute_ssim_elements`, and its C3 parameter
    should be None.

    Parameters
    ----------
    alpha : float
        The range-invariant factor.
    ssim : SSIM
        The SSIM elements.
    return_individual_components : bool, optional
        If True, return the individual components of the SSIM value.

    Returns
    -------
    Union[float, SSIM]
        The SSIM value or the individual components of the SSIM.

    Raises
    ------
    ValueError
        If `ssim.C3` is not None.
    """
    if ssim.C3 is not None:
        return _ssim_from_params_with_C3(alpha, ssim, return_individual_components)

    A1, A2, B1, B2 = (
        2 * alpha * ssim.ux * ssim.uy + ssim.C1,
        2 * alpha * ssim.vxy
        + ssim.C2,  # TOOD here difference with _ssim_from_params_with_C3
        ssim.ux**2 + (alpha**2) * ssim.uy**2 + ssim.C1,
        ssim.vx + (alpha**2) * ssim.vy + ssim.C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D

    if return_individual_components:
        term = 2 * alpha * np.sqrt(ssim.vx * ssim.vy) + ssim.C2
        luminance = A1 / B1
        contrast = term / B2
        structure = A2 / term

        return SSIM(
            SSIM=S,
            luminance=luminance,
            contrast=contrast,
            structure=structure,
            alpha=alpha,
            ux=ssim.ux,
            uy=ssim.uy,
            vx=ssim.vx,
            vy=ssim.vy,
            vxy=ssim.vxy,
            C1=ssim.C1,
            C2=ssim.C2,
        )

    return np.mean(S)


def get_ri_factor(ssim: SSIM) -> float:
    """Compute the range-invariant factor for the RI-SSIM from the SSIM elements.

    Parameters
    ----------
    ssim : SSIMElements
        The SSIM elements, obtained from `ri_ssim.ssim.compute_ssim_elements`.

    Returns
    -------
    float
        The range-invariant factor.
    """

    # create a copy of SSIM without C3
    ssim_without_C3 = copy(ssim)
    ssim_without_C3.C3 = None

    initial_guess = np.array([1])

    res = minimize(
        lambda x: -1 * _ssim_from_params(x), initial_guess, args=ssim_without_C3
    )

    return res.x[0]


# TODO is this important?
def mse_based_range_invariant_structural_similarity(
    target_img: NDArray,
    pred_img: NDArray,
    *,
    win_size: Optional[int] = None,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: bool = False,
    return_individual_components: bool = False,
    **kwargs,
) -> Union[float, SSIM]:
    """MSE based range-invariant structural similarity index.

    Parameters
    ----------
    target_img : np.ndarray
        The target image.
    pred_img : np.ndarray
        The predicted image.
    win_size : int, optional
        The side-length of the sliding window used in comparison. Must be an odd value.
    data_range : float, optional
        The data range of the input image.
    channel_axis : int, optional
        The axis of the image that corresponds to channels.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a normalized
        Gaussian kernel of width sigma=1.5.
    return_individual_components : bool, optional
        If True, return the individual components of the SSIM value.

    Returns
    -------
    float or SSIM
        The MSE-based range-invariant structural similarity index.
    """
    ri_factor = get_mse_based_factor(target_img[None], pred_img[None])

    return range_invariant_structural_similarity(
        target_img,
        pred_img,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        ri_factor=ri_factor,
        return_individual_components=return_individual_components,
        **kwargs,
    )


def range_invariant_structural_similarity(
    target_img: NDArray,
    pred_img: NDArray,
    *,
    win_size: Optional[int] = None,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: Optional[
        bool
    ] = True,  # TODO different default value than skimage
    ri_factor: Optional[float] = None,
    return_individual_components: bool = False,
    **kwargs: dict,
) -> Union[float, SSIM]:
    """Compute the range-invariant structural similarity index between two images.

    # TODO check the definition of the parameters
    Parameters
    ----------
    target_img : numpy.ndarray
        The target image.
    pred_img : numpy.ndarray
        The predicted image.
    win_size : int, optional
        The side-length of the sliding window used in comparison. Must be an odd value.
        If gaussian_weights is True, this is ignored and the window size will depend on
        sigma.
    data_range : float, optional
        The data range of the input image (difference between maximum and minimum
        possible values). By default, this is estimated from the image data type. This
        estimate may be wrong for floating-point image data. Therefore it is recommended
        to always pass this scalar value explicitly.
    channel_axis : int, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds to
        channels.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a normalized
        Gaussian kernel of width sigma=1.5.
    ri_factor : float, optional
        The range-invariant factor.
    return_individual_components : bool, optional
        If True, return the individual components of the SSIM value.
    **kwargs : dict
        Additional keyword arguments, passed to `skimage.metrics.structural_similarity`.

    Returns
    -------
    float or SSIM
        The range-invariant structural similarity index.
    """
    ssim_elements = compute_ssim_elements(
        target_img,
        pred_img,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        **kwargs,
    )

    # retrieve range invariant factor if not provided
    if ri_factor is None:
        ri_factor = get_ri_factor(ssim_elements)

    return _ssim_from_params(ri_factor, ssim_elements, return_individual_components)
