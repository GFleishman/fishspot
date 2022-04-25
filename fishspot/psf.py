import numpy as np
from scipy.ndimage import maximum_filter


def get_context(spot, image, radius):
    """
    """

    p, r = spot, radius
    w = image[
        p[0]-r:p[0]+r+1,
        p[1]-r:p[1]+r+1,
        p[2]-r:p[2]+r+1
    ]
    if np.product(w.shape) != (2*r+1)**3:
        return None
    return w.flatten()


def estimate_psf(image, radius, n_spots=2000, **kwargs):
    """
    """

    # get weak spots
    max_image = maximum_filter(image, size=radius)
    spots = np.column_stack(np.nonzero(image == max_image))
    intensities = image[spots[:, 0], spots[:, 1], spots[:, 2]]
    sort_idxs = np.argsort(intensities)[::-1]
    spots = spots[sort_idxs[:n_spots], :]

    # get context around each weak spot
    blocks = []
    for spot in spots:
        context = get_context(spot, image, radius)
        if context is not None:
            blocks.append(context)
    blocks = np.array(blocks)

    # set ransac filter defaults
    if 'n_propose_model' not in kwargs:
        kwargs['n_propose_model'] = 10
    if 'max_iterations' not in kwargs:
        kwargs['max_iterations'] = 1000
    if 'inlier_threshold' not in kwargs:
        kwargs['inlier_threshold'] = 0.9
    if 'min_inliers' not in kwargs:
        kwargs['min_inliers'] = 25

    # estimate psf with ransac filter
    psf, inliers = ransac(blocks, model=psf_model, **kwargs)
    psf = psf.get_array(radius)

    # return normalized psf
    return psf / np.sum(psf)


class psf_model:

    def __init__(self):
        None

    def _norm_range(self, data):
        mx = np.max(data, axis=1)[..., None]
        mn = np.min(data, axis=1)[..., None]
        return (data - mn) / (mx - mn)

    def _stats(self, data):
        mean = np.mean(data, axis=1, dtype=np.float64)
        std = np.std(data, axis=1, dtype=np.float64)
        return mean, std

    def fit(self, data):
        norm_data = self._norm_range(data)
        self.psf = np.mean(norm_data, axis=0)
        return self

    def get_error(self, data):
        psf_mean, psf_std = self._stats(self.psf[None, ...])
        psf_cent = self.psf - psf_mean
        norm_data = self._norm_range(data)
        dat_mean, dat_std = self._stats(norm_data)
        dat_cent = norm_data - dat_mean[..., None]
        corr = -np.mean(dat_cent*psf_cent, axis=1)/dat_std/psf_std
        return corr.squeeze()

    def get_array(self, radius):
        return np.reshape(self.psf, (2*radius+1,)*3)


def ransac(
    data,
    n_propose_model,
    max_iterations,
    inlier_threshold,
    min_inliers,
    model=psf_model,
    verbose=False,
):
    """
    """

    # initialize
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    psf = model()

    # ensure inlier_threshold is a negative correlation
    if 0 < inlier_threshold < 1:
        inlier_threshold *= -1

    # loop over model proposals
    while iterations < max_iterations:

        # randomly partition data
        all_idxs = np.arange(data.shape[0])
        np.random.shuffle(all_idxs)
        maybe_idxs = all_idxs[:n_propose_model]
        test_idxs = all_idxs[n_propose_model:]
        maybeinliers = data[maybe_idxs]
        test_points = data[test_idxs]

        # fit model, get error
        maybemodel = psf.fit(maybeinliers)
        test_err = psf.get_error(test_points)

        # get fellow inliers
        also_idxs = test_idxs[test_err < inlier_threshold]
        alsoinliers = data[also_idxs]

        # provide feedback if requested
        if verbose:
            print('test_err.min()',test_err.min())
            print('test_err.max()',test_err.max())
            print('np.mean(test_err)',np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers)))

        # update model with new inliers
        if len(alsoinliers) > min_inliers:
            betterdata = np.concatenate(
                (maybeinliers, alsoinliers)
            )
            bettermodel = psf.fit(betterdata)
            better_errs = psf.get_error(betterdata)

            # get updated model error, if best overwrite old model
            thiserr = np.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) )

        # tick tock
        iterations+=1

    # final results
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    return bestfit, {'inliers':best_inlier_idxs}


