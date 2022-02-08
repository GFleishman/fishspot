import numpy as np
from skimage.morphology import white_tophat as skimage_white_tophat
from skimage.exposure import rescale_intensity
from skimage.restoration import richardson_lucy


def white_tophat(image, radius):
    """
    """

    selem = np.ones((radius,)*image.ndim)
    return skimage_white_tophat(image, selem=selem)


def rl_decon(image, psf, **kwargs):
    """
    """

    # normalize image
    norm_image = rescale_intensity(
        image,
        in_range=(0, image.max()),
        out_range=(0, 1),
    )

    # run decon
    return richardson_lucy(norm_image, psf, **kwargs)


def wiener_decon(image, psf, nsr=1e-3):
    """
    """

    # normalize image
    norm_image = rescale_intensity(
        image,
        in_range=(0, image.max()),
        out_range=(0, 1),
    )

    # run decon
    G = np.fft.fftn(norm_image)
    H = np.fft.fftn(psf, s=norm_image.shape)
    H = np.conj(H) / (np.abs(H)**2 + nsr)

    return np.abs(np.fft.ifftn(G * H))
