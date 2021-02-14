from skimage.exposure import rescale_intensity
from skimage.restoration import richardson_lucy


def rl_decon(image, psf, **kwargs):
    """
    """

    norm_image = rescale_intensity(
        image,
        in_range=(0, image.max()),
        out_range=(0, 1),
    )
    return richardson_lucy(norm_image, psf, **kwargs)


