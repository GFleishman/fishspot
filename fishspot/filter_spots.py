import numpy as np
from scipy.spatial import cKDTree


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

