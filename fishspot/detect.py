import numpy as np
from skimage.feature import blob_log


# TODO: potential major improvement - after finding coordinates
#       with LoG filter, template match the PSF to the region around
#       each detected point (Fourier Phase Correlation maybe?).
#       upsample the PSF and data to achieve subvoxel accuracy.


def detect_spots_log(
    image,
    min_radius,
    max_radius,
    num_sigma=5,
    normalize=False,
    **kwargs,
):
    """
    """

    # normalize intensities
    if normalize:
        image = (image - np.mean(image)) / np.std(image)

    # ensure iterable radii
    if not isinstance(min_radius, (tuple, list, np.ndarray)):
        min_radius = (min_radius,)*image.ndim
    if not isinstance(max_radius, (tuple, list, np.ndarray)):
        max_radius = (max_radius,)*image.ndim

    # set given arguments, convert pixel radii to kernel widths
    kwargs['min_sigma'] = np.array(min_radius) / np.sqrt(image.ndim)
    kwargs['max_sigma'] = np.array(max_radius) / np.sqrt(image.ndim)
    kwargs['num_sigma'] = num_sigma

    # set additional defaults
    if 'threshold' not in kwargs and 'threshold_rel' not in kwargs:
        kwargs['threshold'] = 1e-6
    if 'overlap' not in kwargs:
        kwargs['overlap'] = 1.0

    # run, convert back to pixel radii, return
    spots = blob_log(image, **kwargs)
    spots[:, image.ndim:] *= np.sqrt(image.ndim)
    return spots

