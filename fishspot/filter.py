import numpy as np
from skimage.morphology import white_tophat as skimage_white_tophat
from skimage.exposure import rescale_intensity
from skimage.restoration import richardson_lucy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter


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
    mx = image.max()
    norm_image = rescale_intensity(image, in_range=(0, mx), out_range=(0, 1))

    # set some defaults
    if 'num_iter' not in kwargs:
        kwargs['num_iter'] = 20
    if 'clip' not in kwargs:
        kwargs['clip'] = False
    if 'filter_epsilon' not in kwargs:
        kwargs['filter_epsilon'] = 1e-6

    # run decon, renormalize, return
    decon = richardson_lucy(norm_image, psf, **kwargs)
    return rescale_intensity(decon, in_range=(0, 1), out_range=(0, mx))


def apply_foreground_mask(spots, mask, ratio=1):
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
        result = result[result[:, i] >= origin[i]]
        result = result[result[:, i] < origin[i] + span[i]]
    return result


def percentile_filter(spots, percentile):
    """
    """

    thresh = np.percentile(spots[:, -1], percentile)
    return spots[ spots[:, -1] >= thresh ]


def density_filter(spots, radius, neighbor_threshold):
    """
    """

    # get spot-to-spot distances
    tree = cKDTree(spots[:, :3])
    dok = tree.sparse_distance_matrix(tree, radius)
    csr = dok.tocsr()

    # determine loner spots, remove them and return
    density_filter = np.ones(spots.shape[0], dtype=bool)
    for iii in range(spots.shape[0]):
        if csr[iii].count_nonzero() < neighbor_threshold:
            density_filter[iii] = False
    return spots[density_filter]


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


def maximum_deviation_threshold(image, mask=None, winsorize=(1, 99), sigma=8):
    """
    """

    # function to get line from rightmost mode to endpoint
    def get_line(hist, edges):
        peak = np.argmax(hist)
        slope = (hist[peak] - hist[-1]) / (edges[peak] - edges[-1])
        intercept = hist[peak] - slope * edges[peak]
        line = slope * edges[peak:] + intercept
        if np.any(hist[peak+1:-1] > line[1:-1]):  # line should bound histogram
            new_peak, line = get_line(hist[peak+1:], edges[peak+1:])
            return peak + 1 + new_peak, line
        return peak, line

    # a function to get the threshold point from curve and line segment
    def get_point(hist, edges):
        peak, line = get_line(hist, edges)
        line_points = np.vstack((edges[peak:], line)).T
        curve_points = np.vstack((edges[peak:], hist[peak:])).T
        dists = np.min(cdist(curve_points, line_points), axis=1)
        point = np.argmax(dists) + peak
        if np.any(hist[point+1:-1] > hist[point]):  # tail should monotonically decrease
            return peak + 1 + get_point(hist[peak+1:], edges[peak+1])
        return point

    # get histogram, get point, return threshold
    foreground = image[mask > 0] if mask is not None else image
    mn, mx = np.percentile(foreground, winsorize).astype(int)
    hist, edges = np.histogram(foreground, bins=mx - mn, range=(mn, mx), density=True)
    hist = gaussian_filter(hist, sigma=sigma)
    edges = edges[1:]
    return edges[get_point(hist, edges)]

