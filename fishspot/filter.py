import numpy as np
from skimage.morphology import white_tophat as skimage_white_tophat
from skimage.exposure import rescale_intensity
from skimage.restoration import richardson_lucy
from scipy.spatial import cKDTree


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


def apply_foreground_mask(spots, mask, ratio):
    """
    """

    # get spot locations in mask voxel coordinates
    x = np.round(spots[:, :3] * ratio).astype(np.uint16)

    # correct out of range rounding errors
    for i in range(3):
        x[x[:, i] >= mask.shape[i], i] = mask.shape[i] - 1

    # filter spots and return
    return spots[mask[x[:, 0], x[:, 1], x[:, 2]] == 1]


def filter_by_range(spots, origin, span):
    """
    """

    # operate on a copy, filter lower/upper range all axes
    result = np.copy(spots)
    for i in range(3):
        result = result[result[:, i] > origin[i] - 1]
        result = result[result[:, i] < origin[i] + span[i]]
    return result


def remove_duplicates(spots1, spots2, radius, return_duplicate_indices=False):
    """
    """

    # use kd-trees
    tree1 = cKDTree(spots1[:, :3])
    tree2 = cKDTree(spots2[:, :3])

    # search for duplicate pairs
    duplicates = tree1.query_ball_tree(tree2, radius)

    # reformat to lists of row indices
    duplicate_rows1 = []
    duplicate_rows2 = []
    for iii, duplicate_list in enumerate(duplicates):
        if duplicate_list:
            duplicate_rows1.append(iii)
            duplicate_rows2.extend(duplicate_list)
    duplicate_rows2 = list(set(duplicate_rows2))

    # filter arrays
    spots1_filtered = np.delete(spots1, duplicate_rows1, axis=0)
    spots2_filtered = np.delete(spots2, duplicate_rows2, axis=0)

    # return
    if return_duplicate_indices:
        return spots1_filtered, spots2_filtered, duplicate_rows1, duplicate_rows2
    else:
        return spots1_filtered, spots2_filtered
