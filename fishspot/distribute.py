import numpy as np
import dask
import dask.array as da
import dask.delayed as delayed
import ClusterWrap
import time
import fishspot.filter as fs_filter
import fishspot.psf as fs_psf
import fishspot.detect as fs_detect
import fishspot.filter_spots as filter_spots


def distributed_spot_detection(
    array, blocksize,
    white_tophat_args={},
    psf_estimation_args={},
    deconvolution_args={},
    spot_detection_args={},
    cluster_kwargs={},
    mask=None,
    psf=None,
    psf_retries=3,
):
    """
    """

    # set white_tophat defaults
    if 'radius' not in white_tophat_args.keys():
        white_tophat_args['radius'] = 9

    # set psf estimation defaults
    if 'radius' not in psf_estimation_args.keys():
        psf_estimation_args['radius'] = 9

    # set deconvolution defaults
    if 'filter_epsilon' not in deconvolution_args.keys():
        deconvolution_args['filter_epsilon'] = 1e-8

    # set spot detection defaults
    if 'min_blob_radius' not in spot_detection_args.keys():
        spot_detection_args['min_blob_radius'] = 3
    if 'max_blob_radius' not in spot_detection_args.keys():
        spot_detection_args['max_blob_radius'] = 6

    # compute overlap depth
    overlap = 2*spot_detection_args['max_blob_radius']

    # compute mask to array ratio
    if mask is not None:
        ratio = np.array(mask.shape) / array.shape

    # pipeline to run on each block
    def detect_spots_pipeline(block, mask=None, psf=None, block_info=None):

        # get origin (used a few times later)
        origin = np.array(block_info[0]['chunk-location'])
        origin = origin * blocksize - overlap

        # check mask
        if mask is not None:
            mo = np.round(origin * ratio).astype(np.uint16)
            mo = np.maximum(0, mo)
            ms = np.round(blocksize * ratio).astype(np.uint16)
            mask_block = mask[mo[0]:mo[0]+ms[0],
                              mo[1]:mo[1]+ms[1],
                              mo[2]:mo[2]+ms[2],]

            # if there is no foreground, return null result
            if np.sum(mask_block) < 1:
                result = np.empty((1,1,1), dtype=list)
                result[0, 0, 0] = [np.zeros((0, 5)), psf]
                return result

        # background subtraction: white tophat filter
        wth = fs_filter.white_tophat(block, **white_tophat_args)

        if psf is None:
            # automated psf estimation with error handling
            for i in range(psf_retries):
                try:
                    psf = fs_psf.estimate_psf(wth, **psf_estimation_args)
                except ValueError:
                    if 'inlier_threshold' not in psf_estimation_args.keys():
                        psf_estimation_args['inlier_threshold'] = 0.9
                    psf_estimation_args['inlier_threshold'] -= 0.1
                else: break

        # deconvolution
        decon = fs_filter.rl_decon(wth, psf, **deconvolution_args)

        # blob detection
        spots = fs_detect.detect_spots_log(decon, **spot_detection_args)

        # remove spots in the halo
        for i in range(3):
            spots = spots[spots[:, i] > overlap - 1]
            spots = spots[spots[:, i] < overlap + blocksize[i]]

        # adjust for block origin
        if spots.shape[0] > 0:
            spots[:, :3] = spots[:, :3] + origin

        # package and return
        result = np.empty((1,1,1), dtype=list)
        result[0, 0, 0] = [spots, psf]
        return result

    # construct cluster
    with ClusterWrap.cluster(**cluster_kwargs) as cluster:

        # a test dataset as a numpy array
        if isinstance(array, np.ndarray):
            future = cluster.client.scatter(array)
            array_da = da.from_delayed(
                future, shape=array.shape, dtype=array.dtype,
            )
            array_da = array_da.rechunk(blocksize)
            array_da.persist()
            time.sleep(5)  ### a little time for workers to be allocated
            cluster.client.rebalance()

        # a full dataset as a zarr array
        else:
            array_da = da.from_array(array, chunks=blocksize)

        # delay mask
        mask_d = delayed(mask) if mask is not None else None

        # execute spot detection on all blocks
        spots_as_grid = da.map_overlap(
            detect_spots_pipeline, array_da,
            mask=mask_d,
            psf=psf,
            depth=overlap,
            dtype=list,  ### dask generalizes to object
            boundary=0,
            trim=False,
            chunks=(1,1,1),
        ).compute()

    # reformat spots and psf into lists
    spots_list, psf_list = [], []
    for ijk in range(np.prod(spots_as_grid.shape)):
        i, j, k = np.unravel_index(ijk, spots_as_grid.shape)
        spots_list.append(spots_as_grid[i, j, k][0])
        psf_list.append(spots_as_grid[i, j, k][1])

    # concatenate all spots into single array
    spots = np.vstack(spots_list)

    # filter with foreground mask
    if mask is not None:
        spots = filter_spots.apply_foreground_mask(
            spots, mask, ratio,
        )

    # average over all psf estimates
    psf = np.mean(psf_list, axis=0)

    # return results
    return spots, psf

