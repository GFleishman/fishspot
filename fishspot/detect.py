import numpy as np
from cucim.skimage.feature import blob_log as cucim_blob_log
import cupy as cp


# TODO: potential major improvement - after finding coordinates
#       with LoG filter, template match the PSF to the region around
#       each detected point (Fourier Phase Correlation maybe?).
#       upsample the PSF and data to achieve subvoxel accuracy.


def detect_spots_log(
    image,
    min_radius,
    max_radius,
    num_sigma=5,
    **kwargs,
):
    """
    """

    # ensure iterable radii
    if not isinstance(min_radius, (tuple, list, np.ndarray)):
        min_radius = (min_radius,)*image.ndim
    if not isinstance(max_radius, (tuple, list, np.ndarray)):
        max_radius = (max_radius,)*image.ndim

    # compute defaults
    min_radius = np.array(min_radius)
    max_radius = np.array(max_radius)
    min_sigma = 0.8 * min_radius / np.sqrt(image.ndim)
    max_sigma = 1.2 * max_radius / np.sqrt(image.ndim)

    # set given arguments
    kwargs['min_sigma'] = min_sigma
    kwargs['max_sigma'] = max_sigma
    kwargs['num_sigma'] = num_sigma

    # set additional defaults
    if 'overlap' not in kwargs:
        kwargs['overlap'] = 1.0
    if 'threshold' not in kwargs:
        kwargs['threshold'] = None
        kwargs['threshold_rel'] = 0.1

    # print(image.shape)
    # result = []
    # if len(image.shape) == 3:
    #     for i in range(image.shape[0]):
    #         seq = cucim_blob_log(cp.array(image[i,:,:]), **kwargs).get()
    #         seq = np.insert(seq, 0, values=i, axis=1)
    #         result.append(seq)
    #         print(i, len(result), seq.shape)

    # # print(result)
    # return np.concatenate(result, axis=0)
    
    # run white tophat
    i = cp.array(image)
    result = cucim_blob_log(i, **kwargs).get()
    del i
    return result
    
    # run
    # return cucim_blob_log(cp.array(image), **kwargs).get()
