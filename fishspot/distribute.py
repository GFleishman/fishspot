import numpy as np
import dask.delayed as delayed
from ClusterWrap.decorator import cluster
import time
from itertools import product
import fishspot.filter as fs_filter
import fishspot.psf as fs_psf
import fishspot.detect as fs_detect


@cluster
def distributed_spot_detection(
    array, blocksize,
    white_tophat_args={},
    psf_estimation_args={},
    deconvolution_args={},
    spot_detection_args={},
    intensity_threshold=0,
    mask=None,
    psf=None,
    psf_retries=3,
    cluster=None,
    cluster_kwargs={},
):
    """
    """

    # set white_tophat defaults
    if 'radius' not in white_tophat_args:
        white_tophat_args['radius'] = 4

    # set psf estimation defaults
    if 'radius' not in psf_estimation_args:
        psf_estimation_args['radius'] = 9

    # set spot detection defaults
    if 'min_radius' not in spot_detection_args:
        spot_detection_args['min_radius'] = 1
    if 'max_radius' not in spot_detection_args:
        spot_detection_args['max_radius'] = 6

    # compute overlap depth
    all_radii = [white_tophat_args['radius'],
                 psf_estimation_args['radius'],
                 spot_detection_args['max_radius'],]
    overlap = int(2*max(np.max(x) for x in all_radii))

    # don't detect spots in the overlap region
    if 'exclude_border' not in spot_detection_args:
        spot_detection_args['exclude_border'] = overlap

    # compute mask to array ratio
    if mask is not None:
        ratio = np.array(mask.shape) / array.shape
        stride = np.round(blocksize * ratio).astype(int)

    # compute number of blocks
    nblocks = np.ceil(np.array(array.shape) / blocksize).astype(int)

    # determine indices for blocking
    indices, psfs = [], []
    for (i, j, k) in product(*[range(x) for x in nblocks]):
        start = np.array(blocksize) * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(array.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        # determine if block is in foreground
        if mask is not None:
            mo = stride * (i, j, k)
            mask_slice = tuple(slice(x, x+y) for x, y in zip(mo, stride))
            if not np.any(mask[mask_slice]): continue

        # add coordinate slices to list
        indices.append(coords)
        psfs.append(psf)

    # pipeline to run on each block
    def detect_spots_pipeline(coords, psf):

        # load data, background subtract, deconvolve, detect blobs
        block = array[coords]
        wth = fs_filter.white_tophat(block, **white_tophat_args)
        if psf is None:
            # automated psf estimation with error handling
            for i in range(psf_retries):
                try:
                    psf = fs_psf.estimate_psf(wth, **psf_estimation_args)
                except ValueError:
                    if 'inlier_threshold' not in psf_estimation_args:
                        psf_estimation_args['inlier_threshold'] = 0.9
                    psf_estimation_args['inlier_threshold'] -= 0.1
                else: break
        decon = fs_filter.rl_decon(wth, psf, **deconvolution_args)
        spots = fs_detect.detect_spots_log(decon, **spot_detection_args)

        # if no spots are found, ensure consistent format
        if spots.shape[0] == 0:
            return np.zeros((0, 7)), psf
        else:
            # append image intensities
            spot_coords = spots[:, :3].astype(int)
            intensities = block[spot_coords[:, 0], spot_coords[:, 1], spot_coords[:, 2]]
            spots = np.concatenate((spots, intensities[..., None]), axis=1)
            spots = spots[ spots[..., -1] > intensity_threshold ]

            # adjust for block origin
            origin = np.array([x.start for x in coords])
            spots[:, :3] = spots[:, :3] + origin
            return spots, psf
    # END: CLOSURE

    # wait for at least one worker to be fully instantiated
    while ((cluster.client.status == "running") and
           (len(cluster.client.scheduler_info()["workers"]) < 1)):
        time.sleep(1.0)

    # submit all alignments to cluster
    spots_and_psfs = cluster.client.gather(
        cluster.client.map(detect_spots_pipeline, indices, psfs)
    )

    # reformat to single array of spots and single psf
    spots, psfs = [], []
    for x, y in spots_and_psfs:
        spots.append(x)
        psfs.append(y)
    spots = np.vstack(spots)
    psf = np.mean(psfs, axis=0)

    # filter with foreground mask
    if mask is not None:
        spots = fs_filter.apply_foreground_mask(
            spots, mask, ratio,
        )

    # return results
    return spots, psf

