import numpy as np
from skimage.morphology import white_tophat as skimage_white_tophat


def white_tophat(image, radius):
    """
    """

    selem = np.ones((radius,)*image.ndim)
    return skimage_white_tophat(image, selem=selem)


