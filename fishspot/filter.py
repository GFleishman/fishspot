import logging
import numpy as np

from skimage.morphology import white_tophat as skimage_white_tophat
from skimage.exposure import rescale_intensity
from skimage.restoration import richardson_lucy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter


logger = logging.getLogger(__name__)


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
    mn, mx = image.min(), image.max()
    norm_image = rescale_intensity(image, in_range=(mn, mx), out_range=(0, 1))

    # set some defaults
    if 'num_iter' not in kwargs:
        kwargs['num_iter'] = 20
    if 'clip' not in kwargs:
        kwargs['clip'] = False
    if 'filter_epsilon' not in kwargs:
        kwargs['filter_epsilon'] = 1e-6

    # pad the input based on psf size
    pad = tuple((x//2 + 2,)*2 for x in psf.shape)
    norm_image = np.pad(norm_image, pad, mode='reflect')

    # run decon, renormalize, return
    decon = richardson_lucy(norm_image, psf, **kwargs)

    # remove the pad
    crop = tuple(slice(x[0], -x[1]) for x in pad)
    decon = decon[crop]

    return rescale_intensity(decon, in_range=(0, 1), out_range=(mn, mx))


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


def density_filter(
    spots, radius, neighbor_threshold,
    weight_by_intensity=False,
    weight_by_size=False,
    inverted=False
):
    """
    """

    # get neighbors lists
    tree = cKDTree(spots[:, :3])
    neighbors = tree.query_ball_tree(tree, radius)

    # get count per spot, weight by intensity and/or size
    counts = np.ones(spots.shape[0], dtype=np.uint16)
    if weight_by_intensity:
        intensities = spots[:, -1]
        counts = counts * intensities / np.median(intensities)
    if weight_by_size:
        volumes = np.prod(spots[:, 3:6], axis=1)
        counts = counts * volumes / np.min(volumes)

    # check each point for sufficient neighborhood density
    density_filter = np.ones(spots.shape[0], dtype=bool)
    for iii, neighbors_list in enumerate(neighbors):
        if sum([counts[x] for x in neighbors_list]) < neighbor_threshold:
            density_filter[iii] = False
    if inverted:
        density_filter = ~density_filter
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


def maximum_deviation_threshold(image, mask=None, winsorize=(1, 99), sigma=8., max_steps=1000):
    """
    Select a threshold for a unimodal histogram with the maximum deviation method.
    This method draws a straight line from the peak/mode of the histogram to the
    tail and then finds the point on the histogram between the peak and tail which
    is maximally distant from this straight line.

    Here the unimodal assumption is relaxed in two ways: the input histogram is
    smoothed to remove high frequency noise and the peak is chosen to be the
    rightmost mode of the histogram. That is the histogram may have more than one
    mode, but it is assumed that there is a gradually decreasing tail to the right
    of the final mode.

    Parameters
    ----------
    image : nd-array
        The data which we want to threshold

    mask : nd-array (default: None)
        A binary mask of image. Only voxels in the foreground will be considered.

    winsorize : length 2 tuple (default: (1, 99))
        The lower and upper percentiles to cut off from the histogram. This adds
        robustness to outliers.

    sigma : float (default: 8.)
        The standard deviation of the gaussian applied to the histogram to smooth
        high frequency components (extra modes/bumps due to noise).

    max_steps : int (default: 1000)
        Safety cap on the number of get_line/get_point iterations. Each iteration
        shrinks the working histogram by at least one bin, so convergence within
        len(histogram) steps is guaranteed; this cap just bounds the worst-case
        run time for very large or pathological histograms. If the cap is hit,
        -1 is returned instead of a threshold.

    Returns
    -------
        threshold : float
            The maximum deviation threshold for the rightmost mode to tail of the
            image histogram. -1 if no threshold could be determined.
    """

    # function to get line from rightmost mode to endpoint
    # iterative, not recursive, and capped at max_steps: each step shrinks
    # hist/edges by at least one element, so this is guaranteed to terminate
    # within len(hist) steps, but the recursive version this replaced could
    # blow Python's call stack on histograms needing hundreds of steps to
    # bound (e.g. a wide, noisy or multi-modal intensity histogram).
    def get_line(hist, edges):
        offset = 0
        for _ in range(max_steps):
            peak = np.argmax(hist)
            slope = (hist[peak] - hist[-1]) / (edges[peak] - edges[-1])
            intercept = hist[peak] - slope * edges[peak]
            line = slope * edges[peak:] + intercept
            if not np.any(hist[peak+1:-1] > line[1:-1]):  # line should bound histogram
                return offset + peak, line
            offset += peak + 1
            hist = hist[peak+1:]
            edges = edges[peak+1:]
        return None, None

    # a function to get the threshold point from curve and line segment
    def get_point(hist, edges):
        offset = 0
        for _ in range(max_steps):
            peak, line = get_line(hist, edges)
            if peak is None:
                return None
            line_points = np.vstack((edges[peak:], line)).T
            curve_points = np.vstack((edges[peak:], hist[peak:])).T
            dists = np.min(cdist(curve_points, line_points), axis=1)
            point = np.argmax(dists) + peak
            if np.all(hist[point+1:-1] <= hist[point]):  # tail should monotonically decrease
                return offset + point
            offset += peak + 1
            hist = hist[peak+1:]
            edges = edges[peak+1:]
        return None

    # get histogram, get point, return threshold
    foreground = image[mask > 0] if mask is not None else image
    mn, mx = np.percentile(foreground, winsorize).astype(int)
    logger.info(f'Foreground min/max: {mn}/{mx}')
    if mx <= mn:
        # this can apparently happen when blocks with no foreground info are passed to the worker
        return -1
    hist, edges = np.histogram(foreground, bins=mx-mn, range=(mn, mx), density=True)
    hist = gaussian_filter(hist, sigma=sigma)
    edges = edges[1:]
    point = get_point(hist, edges)
    if point is None:
        logger.info(f'Peak search did not converge within {max_steps} steps; no threshold found')
        return -1
    return edges[point]
