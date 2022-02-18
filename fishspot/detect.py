import numpy as np
from skimage.feature import blob_log


# TODO: potential major improvement - after finding coordinates
#       with LoG filter, template match the PSF to the region around
#       each detected point (Fourier Phase Correlation maybe?).
#       upsample the PSF and data to achieve subvoxel accuracy.


def detect_spots_log(
    image,
    min_blob_radius,
    max_blob_radius,
    **kwargs,
):
    """
    """

    # compute defaults
    min_blob_radius = np.array(min_blob_radius)
    max_blob_radius = np.array(max_blob_radius)
    min_sigma = 0.8 * min_blob_radius / np.sqrt(3)
    max_sigma = 1.2 * max_blob_radius / np.sqrt(3)
    num_sigma = int(np.ceil(np.max(max_blob_radius - min_blob_radius)))

    # set defaults
    if 'min_sigma' not in kwargs:
        kwargs['min_sigma'] = min_sigma
    if 'max_sigma' not in kwargs:
        kwargs['max_sigma'] = max_sigma
    if 'num_sigma' not in kwargs:
        kwargs['num_sigma'] = num_sigma
    if 'overlap' not in kwargs:
        kwargs['overlap'] = 0.8

    # run
    return blob_log(image, **kwargs)

