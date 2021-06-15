import numpy as np


def apply_foreground_mask(spots, mask, mask_spacing):
    """
    """

    # get spot locations in mask voxel coordinates
    x = np.round(spots/mask_spacing).astype(np.uint16)

    # correct out of range rounding errors
    for i in range(3):
        x[x[:, i] >= mask.shape[i], i] = mask.shape[i] - 1

    # filter spots and return
    return spots[mask[x[:, 0], x[:, 1], x[:, 2]] == 1]


