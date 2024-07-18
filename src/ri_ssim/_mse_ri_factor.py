"""
Computes PSNR of a batch of monochrome images.
NOTE that a numpy version and np.Tensor version have slightly different values.
e9b29ba0b21f3b5fbd0f915309dcd18ecfee0f55
"""

import numpy as np


def _get_factor(gt, x):
    factor = np.sum(gt * x, axis=1, keepdims=True) / (
        np.sum(x * x, axis=1, keepdims=True)
    )
    return factor


def get_mse_based_factor(gt, pred):
    """
    Adapted from https://github.com/juglab/ScaleInvPSNR/blob/master/psnr.py
    It rescales the prediction to ensure that the prediction has the same range as the ground truth.
    """
    gt = gt.reshape(len(gt), -1)
    pred = pred.reshape(len(gt), -1)
    return _get_factor(gt, pred)
