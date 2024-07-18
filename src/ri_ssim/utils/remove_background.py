"""Background removal function."""

import numpy as np
from numpy.typing import NDArray


def remove_background(x: NDArray, pmin: float = 3, dtype: type = np.float32) -> NDArray:
    """Remove the background from an image using percentile.

    Parameters
    ----------
    x : numpy.ndarray
        The input image.
    pmin : float
        The percentile to remove from the image.
    dtype : type
        The data type of the output image.

    Returns
    -------
    numpy.ndarray
        The image with the background removed.
    """
    mi = np.percentile(x, pmin, keepdims=True)

    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        x = x - mi

    return x
