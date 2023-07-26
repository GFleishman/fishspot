import numpy as np
import dask.delayed as delayed
from ClusterWrap.decorator import cluster
import time
from itertools import product
import fishspot.filter as fs_filter
import fishspot.psf as fs_psf
import fishspot.detect as fs_detect
from fishspot.assign import gravity_flow
import tempfile
import os
import zarr
from numcodecs import Blosc


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


@cluster
def distributed_gravity_flow(
    spots,
    masks,
    spacing,
    iterations,
    learning_rate,
    max_displacement,
    blocksize,
    mask_density=1.0,
    foreground_mask=None,
    temporary_directory=None,
    cluster=None,
    cluster_kwargs={},
    callback=None,
):
    """
    Run the gravity_flow algorithm on blocks from a larger image.
    Blocks overlap to ensure all spots have sufficient context.

        Parameters
    ----------
    spots : 2d array, Nxd for N spots in d dimensions
        The spots we want to assign

    masks : arraylike (zarr or numpy array)
        The multi-integer map of masks we want to assign spots to

    spacing : 1d-array
        The voxel spacing of the masks image

    iterations : int
        The number of iterations to run the flow

    learning_rate : float
        At each iteration the spot displacements will be scaled such that the
        largest displacement of any spot will be equal to this number in micrometers

    max_displacement : float
        No spot will be allowed to displace larger than this amount in micrometers

    blocksize : tuple
        The number of voxels per axis that independent blocks should be

    mask_density : float (default: 1.0)
        A multiplier applied to the mask distance transform. If you want to increase
        the mask-to-spot forces relative to the spot-to-spot forces, increase this number.
        Alternatively if you want to decrease the mask-to-spot forces relative to the
        spot-to-spot forces, decrease this number. Mask-to-spot forces promote assignment
        and spot-to-spot forces promote clustering.

    foreground_mask : ndarray (default: None)
        A binary mask indicating which region of masks is foreground and requires
        spot assignment. Does not need to be the same voxel dimensions as masks,
        but should have the same physical domain.

    temporary_directory : string (default: None)
        Temporary files may be created. The temporary files will be in their own
        folder within the `temporary_directory`. The default is the current
        directory. Temporary files are removed if the function completes
        successfully.

    cluster : ClusterWrap.cluster object (default: None)
        Only set if you have constructed your own static cluster. The default behavior
        is to construct a cluster for the duration of this function, then close it
        when the function is finished.

    cluster_kwargs : dict (default: {})
        Arguments passed to ClusterWrap.cluster
        If working with an LSF cluster, this will be
        ClusterWrap.janelia_lsf_cluster. If on a workstation
        this will be ClusterWrap.local_cluster.
        This is how distribution parameters are specified.

    callback : function (default: None)
        Any function you want to be run at the end of each iteration.

    Returns
    -------
    counts : dict
        The number of spots assigned to each mask segment. The segment indices are keys
        and the values are the number of spots assigned to that segment index.

    assignments : 1d-array
        The index of the segment each spot was assigned to (0 is unassigned/background).
        This array is parallel to the input spots array, meaning assignments[iii]
        corresponds to spots[iii]
    """

    # create temporary directory, save spots there
    temporary_directory = tempfile.TemporaryDirectory(
        prefix='.', dir=temporary_directory or os.getcwd(),
    )
    np.save(temporary_directory.name + '/spots.npy', spots)

    # ensure masks are a zarr array
    if not isinstance(masks, zarr.Array):
        masks_zarr_path = temporary_directory.name + '/masks.zarr'
        masks_zarr = zarr.open(
            masks_zarr_path, 'w',
            shape=masks.shape,
            chunks=blocksize,
            dtype=masks.dtype,
            compressor=Blosc(cname='zstd', clevel=4, shuffle=Blosc.BITSHUFFLE),
        )
        masks_zarr[...] = masks
    else: masks_zarr = masks

    # determine image slices for blocking
    blocksize = np.array(blocksize)
    nblocks = np.ceil(np.array(masks.shape) / blocksize).astype(int)
    overlaps = np.ceil(2 * max_displacement / spacing).astype(int)
    indices, slices = [], []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blocksize * (i, j, k) - overlaps
        stop = start + blocksize + 2 * overlaps
        start = np.maximum(0, start)
        stop = np.minimum(masks.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        foreground = True
        if foreground_mask is not None:
            start = blocksize * (i, j, k)
            stop = start + blocksize
            ratio = np.array(foreground_mask.shape) / masks.shape
            start = np.round( ratio * start ).astype(int)
            stop = np.round( ratio * stop ).astype(int)
            foreground_crop = foreground_mask[tuple(slice(x, y) for x, y in zip(start, stop))]
            if not np.any(foreground_crop): foreground = False

        if foreground:
            indices.append((i, j, k))
            slices.append(coords)

    # closure for spot detection function
    def gravity_flow_block(index, slices):

        # print some feedback
        print("Block index: ", index, "\nSlices: ", slices, flush=True)

        # read the mask and spots data
        masks = masks_zarr[slices]
        spots = np.load(temporary_directory.name + '/spots.npy')

        # get all spots for the roi
        start = [x.start + 1 for x in slices] * spacing
        stop = [x.stop - 2 for x in slices] * spacing
        included = np.all(spots >= start, axis=1) * np.all(spots <= stop, axis=1)
        spots = spots[included]

        # flag spots that are in the overlap regions
        start = blocksize * index * spacing
        stop = start + blocksize * spacing
        outer_spots = np.any((spots < start) + (spots >= stop), axis=1)

        # handle blocks that have no interior spots
        if len(spots[~outer_spots]) == 0:
            print("No spots, returning default", flush=True)
            return {x:0 for x in np.unique(masks)}, {}

        # handle blocks that have no masks
        assigned_indices = np.nonzero(included)[0][~outer_spots]
        if len(np.unique(masks)) == 1:
            print("No masks, returning default", flush=True)
            return {}, {x:0 for x in assigned_indices}

        # rebase spot coordinates to cropped origin
        origin = [x.start for x in slices] * spacing
        spots = spots - origin

        # print some feedback
        print("Number of spots: ", spots.shape[0], flush=True)

        # run spot assignment
        counts, raw_assignments = gravity_flow(
            spots,
            masks,
            spacing,
            iterations,
            learning_rate,
            max_displacement,
            mask_density=mask_density,
            callback=callback,
        )

        # print some feedback
        print("COMPLETE\n", flush=True)

        # remove overlap spots from result and reformat assignments
        for osa in raw_assignments[outer_spots]: counts[osa] -= 1
        raw_assignemnts = raw_assignments[~outer_spots]
        assignments = {x:y for x, y in zip(assigned_indices, raw_assignments)}

        # return
        return counts, assignments

    # submit all alignments to cluster
    results = cluster.client.gather(cluster.client.map(gravity_flow_block, indices, slices))

    # merge all results
    counts, assignments = results.pop(0)
    for more_counts, more_assignments in results:
        for index in more_counts.keys():
            if index in counts.keys(): counts[index] += more_counts[index]
            else: counts[index] = more_counts[index]
        assignments = {**assignments, **more_assignments}

    # return
    return counts, assignments

