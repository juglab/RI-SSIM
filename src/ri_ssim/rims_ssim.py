"""
Multi-scale Microscopy Structural Similarity Index (MS-MicroSSIM) implementation
"""

from typing import Optional, Tuple, Union

# TODO can we remove dependency on torch?
import torch
from numpy.typing import NDArray
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from .ri_ssim import get_ri_factor
from .ssim import compute_ssim_elements


def range_invariant_multiscale_structural_similarity(
    target_img: NDArray,
    pred_img: NDArray,
    *,
    betas: tuple[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    win_size: Optional[int] = None,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: bool = False,
    ri_factor: Optional[float] = None,
    return_ri_factor: bool = False,
    ri_factor_kwargs: dict = {},
    ms_ssim_kwargs: dict = {},
    **kwargs,
) -> Union[float, Tuple[float, float]]:
    """Range invariant multiscale SSIM between two images.

    # TODO description

    Parameters
    ----------
    target_img : NDArray
        The ground truth image.
    pred_img : NDArray
        The predicted image.
    betas : tuple[float]
        The weights for each scale.
    win_size : Optional[int], optional
        The size of the sliding window, by default None.
    data_range : Optional[float], optional
        The range of the data, by default None.
    channel_axis : Optional[int], optional
        The channel axis, by default None.
    gaussian_weights : bool, optional
        Use gaussian weights, by default False.
    ri_factor : Optional[float], optional
        The range invariant factor, by default None.
    return_ri_factor : bool, optional
        Whether to return the range invariant factor, by default False.
    ri_factor_kwargs : dict, optional
        The parameters for the range invariant factor computation.
    ms_ssim_kwargs : dict, optional
        The parameters for the MS-SSIM computation.
    **kwargs : dict
        Additional parameters.

    Returns
    -------
    float or (float, float)
        The MS-SSIM value or a tuple with the MS-SSIM value and the range invariant factor.
    """

    if ri_factor is None:
        ssim_dict = compute_ssim_elements(
            target_img,
            pred_img,
            win_size=win_size,
            data_range=data_range,
            channel_axis=channel_axis,
            gaussian_weights=gaussian_weights,
            **ri_factor_kwargs,
        )
        ri_factor = get_ri_factor(ssim_dict)

    # scale image
    pred_img = pred_img * ri_factor

    # convert to torch
    gt_torch = torch.Tensor(target_img[None, None] * 1.0)
    pred_torch = torch.Tensor(pred_img[None, None] * 1.0)

    # run torch MS-SSIM
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=data_range,
        gaussian_kernel=gaussian_weights,
        betas=betas,
        **ms_ssim_kwargs,
    )

    if return_ri_factor:
        return ms_ssim(pred_torch, gt_torch), ri_factor

    return ms_ssim(pred_torch, gt_torch)
