"""Transformation parameters."""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm  # TODO is tqd really necessary for the metrics?

from ..ri_ssim import _ssim_from_params
from ..ssim import compute_ssim_elements


# TODO is this used anywhere?
def get_transformation_params(gt, pred, **ssim_kwargs):
    ux_arr = []
    uy_arr = []
    vx_arr = []
    vy_arr = []
    vxy_arr = []

    for idx in tqdm(range(len(gt))):
        gt_tmp = gt[idx]
        pred_tmp = pred[idx]

        ssim_dict = compute_ssim_elements(
            gt_tmp,
            pred_tmp,
            data_range=gt_tmp.max() - gt_tmp.min(),
            return_individual_components=True,
            **ssim_kwargs,
        )
        ux, uy, vx, vy, vxy, C1, C2 = (
            ssim_dict["ux"],
            ssim_dict["uy"],
            ssim_dict["vx"],
            ssim_dict["vy"],
            ssim_dict["vxy"],
            ssim_dict["C1"],
            ssim_dict["C2"],
        )
        ux_arr.append(ux)
        uy_arr.append(uy)
        vx_arr.append(vx)
        vy_arr.append(vy)
        vxy_arr.append(vxy)

    ux = np.concatenate(ux_arr, axis=0)
    uy = np.concatenate(uy_arr, axis=0)
    vx = np.concatenate(vx_arr, axis=0)
    vy = np.concatenate(vy_arr, axis=0)
    vxy = np.concatenate(vxy_arr, axis=0)

    other_args = (
        ux,
        uy,
        vx,
        vy,
        vxy,
        C1,
        C2,
    )

    initial_guess = np.array([1])
    res = minimize(
        lambda *args: -1 * _ssim_from_params(*args), initial_guess, args=other_args
    )
    return res.x[0]
