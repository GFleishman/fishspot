import numpy as np


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


