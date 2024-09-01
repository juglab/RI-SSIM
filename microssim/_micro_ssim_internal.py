from typing import Dict, Union

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from ._mse_ri_factor import get_mse_based_factor
from ._ssim_raw import structural_similarity_dict


def _ssim_from_params_with_C3(
    alpha, ux, uy, vx, vy, vxy, C1, C2, C3=None, return_individual_components=False
):
    lum_num = 2 * alpha * ux * uy + C1
    lum_denom = ux**2 + (alpha**2) * uy**2 + C1

    contrast_num = 2 * alpha * np.sqrt(vx * vy) + C2
    contrast_denom = vx + (alpha**2) * vy + C2

    structure_denom = alpha * np.sqrt(vx * vy) + C3
    structure_num = alpha * vxy + C3

    num = lum_num * contrast_num * structure_num
    denom = lum_denom * contrast_denom * structure_denom
    S = num / denom
    if return_individual_components:
        return {
            "SSIM": S,
            "luminance": lum_num / lum_denom,
            "contrast": contrast_num / contrast_denom,
            "structure": structure_num / structure_denom,
            "alpha": alpha,
            "ux": ux,
            "uy": uy,
            "vx": vx,
            "vy": vy,
            "vxy": vxy,
            "C1": C1,
            "C2": C2,
            "C3": C3,
        }
    return np.mean(S)


def _ssim_from_params(
    alpha, ux, uy, vx, vy, vxy, C1, C2, C3=None, return_individual_components=False
):
    if C3 is not None:
        return _ssim_from_params_with_C3(
            alpha,
            ux,
            uy,
            vx,
            vy,
            vxy,
            C1,
            C2,
            C3=C3,
            return_individual_components=return_individual_components,
        )

    A1, A2, B1, B2 = (
        2 * alpha * ux * uy + C1,
        2 * alpha * vxy + C2,
        ux**2 + (alpha**2) * uy**2 + C1,
        vx + (alpha**2) * vy + C2,
    )
    D = B1 * B2
    S = (A1 * A2) / D

    if return_individual_components:
        term = 2 * alpha * np.sqrt(vx * vy) + C2
        luminance = A1 / B1
        contrast = term / B2
        structure = A2 / term
        return {
            "SSIM": S,
            "luminance": luminance,
            "contrast": contrast,
            "structure": structure,
            "alpha": alpha,
            "ux": ux,
            "uy": uy,
            "vx": vx,
            "vy": vy,
            "vxy": vxy,
            "C1": C1,
            "C2": C2,
        }

    return np.mean(S)


def get_ri_factor(ssim_dict: Dict[str, np.ndarray]):
    other_args = (
        ssim_dict["ux"],
        ssim_dict["uy"],
        ssim_dict["vx"],
        ssim_dict["vy"],
        ssim_dict["vxy"],
        ssim_dict["C1"],
        ssim_dict["C2"],
    )
    initial_guess = np.array([1])
    res = minimize(
        lambda *args: -1 * _ssim_from_params(*args), initial_guess, args=other_args
    )
    return res.x[0]


def get_transformation_params(gt, pred, **ssim_kwargs):
    ux_arr = []
    uy_arr = []
    vx_arr = []
    vy_arr = []
    vxy_arr = []

    for idx in tqdm(range(len(gt))):
        gt_tmp = gt[idx]
        pred_tmp = pred[idx]

        ssim_dict = structural_similarity_dict(
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
        # reshape allows handling differently sized images.
        ux_arr.append(
            ux.reshape(
                -1,
            )
        )
        uy_arr.append(
            uy.reshape(
                -1,
            )
        )
        vx_arr.append(
            vx.reshape(
                -1,
            )
        )
        vy_arr.append(
            vy.reshape(
                -1,
            )
        )
        vxy_arr.append(
            vxy.reshape(
                -1,
            )
        )

    ux = np.concatenate(ux_arr)
    uy = np.concatenate(uy_arr)
    vx = np.concatenate(vx_arr)
    vy = np.concatenate(vy_arr)
    vxy = np.concatenate(vxy_arr)

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

    # initial_guess = np.array([1])
    # res = minimize(
    #     lambda *args: -1 * _contrast_sensitivity_from_params(*args),
    #     initial_guess,
    #     args=other_args,
    # )
    # alpha = res.x[0]
    # initial_guess = np.array([0])
    # new_args = (alpha,) + other_args
    # res = minimize(
    #     lambda *args: -1 * _luminance_from_params(*args), initial_guess, args=new_args
    # )
    # offset = res.x[0]
    # return alpha, offset


def mse_based_range_invariant_structural_similarity(
    target_img,
    pred_img,
    *,
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    return_individual_components=False,
    **kwargs,
):
    ri_factor = get_mse_based_factor(target_img[None], pred_img[None])

    return micro_SSIM(
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


def micro_SSIM(
    target_img,
    pred_img,
    *,
    win_size=None,
    data_range=None,
    channel_axis=None,
    gaussian_weights=True,
    ri_factor: Union[float, None] = None,
    return_individual_components: bool = False,
    **kwargs,
):
    ssim_dict = structural_similarity_dict(
        target_img,
        pred_img,
        win_size=win_size,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        **kwargs,
    )
    if ri_factor is None:
        ri_factor = get_ri_factor(ssim_dict)
    ux, uy, vx, vy, vxy, C1, C2, C3 = (
        ssim_dict["ux"],
        ssim_dict["uy"],
        ssim_dict["vx"],
        ssim_dict["vy"],
        ssim_dict["vxy"],
        ssim_dict["C1"],
        ssim_dict["C2"],
        ssim_dict["C3"],
    )

    return _ssim_from_params(
        ri_factor,
        ux,
        uy,
        vx,
        vy,
        vxy,
        C1,
        C2,
        C3=C3,
        return_individual_components=return_individual_components,
    )


def remove_background(x, pmin=3, dtype=np.float32):
    mi = np.percentile(x, pmin, keepdims=True)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        x = x - mi
    return x


# if __name__ == "__main__":
#     import numpy as np
#     from skimage.io import imread

#     def load_tiff(path):
#         """
#         Returns a 4d numpy array: num_imgs*h*w*num_channels
#         """
#         data = imread(path, plugin="tifffile")
#         return data

#     img1 = load_tiff(
#         "/group/jug/ashesh/data/paper_stats/Test_P64_G32_M50_Sk8/gt_D21.tif"
#     )
#     img2 = load_tiff(
#         "/group/jug/ashesh/data/paper_stats/Test_P64_G32_M50_Sk8/pred_training_disentangle_2404_D21-M3-S0-L8_1.tif"
#     )
#     ch_idx = 0
#     img_gt = img1[0, ..., ch_idx]
#     img_pred = img2[0, ..., ch_idx]
#     print(
#         "SSIM",
#         range_invariant_structural_similarity(
#             img_gt,
#             img_pred,
#             data_range=img_gt.max() - img_gt.min(),
#             ri_factor=1.0,
#         ),
#     )

#     print(
#         "RI-SSIM",
#         range_invariant_structural_similarity(
#             img_gt,
#             img_pred,
#             data_range=img_gt.max() - img_gt.min(),
#         ),
#     )

#     print(
#         "RI-SSIM using MSE based:",
#         mse_based_range_invariant_structural_similarity(
#             img_gt,
#             img_pred,
#             data_range=img_gt.max() - img_gt.min(),
#         ),
#     )