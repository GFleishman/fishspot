import numpy as np
from skimage.morphology import white_tophat as skimage_white_tophat
from skimage.exposure import rescale_intensity
from skimage.restoration import richardson_lucy


def white_tophat(image, radius):
    """
    """

    # ensure iterable radius
    if not isinstance(radius, (tuple, list, np.ndarray)):
        radius = (radius,)*image.ndim

    # convert to footprint shape
    shape = [2*r+1 for r in radius]

    # run white tophat
    return skimage_white_tophat(image, footprint=np.ones(shape))


def rl_decon(image, psf, **kwargs):
    """
    """

    # normalize image
    norm_image = rescale_intensity(
        image,
        in_range=(0, image.max()),
        out_range=(0, 1),
    )

    # set some defaults
    if 'num_iter' not in kwargs:
        kwargs['num_iter'] = 20
    if 'clip' not in kwargs:
        kwargs['clip'] = False
    if 'filter_epsilon' not in kwargs:
        kwargs['filter_epsilon'] = 1e-6

    # run decon
    return richardson_lucy(norm_image, psf, **kwargs)

