import numpy as np
import SimpleITK as sitk
from scipy.ndimage import find_objects
from scipy.ndimage import shift
from scipy.ndimage import spline_filter
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter


def gravity_flow(
    spots,
    masks,
    spacing,
    iterations,
    learning_rate,
    max_displacement,
    sigma=2.0,
    mask_density=1.0,
    callback=None,
):
    """
    Assign coordinate data to segment data by flowing coordinates along gravity gradients

    Parameters
    ----------
    spots : 2d array, Nxd for N spots in d dimensions
        The spots we want to assign

    masks : nd-array
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

    mask_density : float (default: 1.0)
        A multiplier applied to the mask distance transform. If you want to increase
        the mask-to-spot forces relative to the spot-to-spot forces, increase this number.
        Alternatively if you want to decrease the mask-to-spot forces relative to the
        spot-to-spot forces, decrease this number. Mask-to-spot forces promote assignment
        and spot-to-spot forces promote clustering.

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

    # the constant mask density
    masks_binary = masks > 0
    masks_edt = distance_transform_edt(masks, sampling=spacing)

    # unit mass
    unit_mass = np.zeros((3,) * masks.ndim)
    unit_mass[(slice(1, 2),) * masks.ndim] = 1
    unit_mass = spline_filter(unit_mass)

    # spots in voxel units
    coords = spots[:, :3] / spacing
    coords_updated = np.copy(coords)
    forces = np.empty_like(coords_updated)

    # construct the linear filter
    center = []
    for x in np.array(masks.shape) / 2:
        if x % 1 == 0: center.append(slice(int(x)-1, int(x)+1))
        else: center.append(slice(int(x), int(x)+1))
    center = tuple(center)
    linear_filter = np.ones(masks.shape, dtype=np.float64)
    linear_filter[center] = 0
    linear_filter = distance_transform_edt(linear_filter, sampling=spacing)**-1
    x, y = np.partition(np.unique(linear_filter), -3)[-3:-1]
    linear_filter[center] = 2*y - x
    linear_filter_fft = np.fft.rfftn(np.fft.fftshift(linear_filter))

    # convert max_displacement to voxel units
    max_displacement = np.mean(max_displacement / spacing)

    # begin flow
    for iii in range(iterations):

        # get density
        density = mask_density * masks_edt
        centers = np.around(coords_updated).astype(int)
        deltas = coords_updated - centers
        for center, delta in zip(centers, deltas):
            shifted_mass = shift(unit_mass, delta, prefilter=False)
            density[tuple(slice(c-1, c+2) for c in center)] += shifted_mass

        # convert to potential, then forces
#        potential = np.fft.irfftn( np.fft.rfftn(density) * linear_filter_fft, density.shape)
        potential = gaussian_filter(density, sigma / spacing)
        force_field = np.array(np.gradient(potential, *spacing)).transpose(1,2,3,0)
        for jjj in range(3):
            forces[..., jjj] = map_coordinates(force_field[..., jjj], coords_updated.transpose())

        # scale forces, convert to voxel units, truncate total displacements
        forces_mag = np.linalg.norm(forces, axis=-1)
        forces = forces * (learning_rate / np.mean(forces_mag) / spacing)
        displacements = coords_updated + forces
        too_far = np.linalg.norm(displacements - coords, axis=-1) > max_displacement
        forces[too_far] = 0
        out_of_bounds = np.any(displacements < 1, axis=1)
        out_of_bounds += np.any(displacements > (np.array(density.shape) - 2), axis=1)
        forces[out_of_bounds] = 0

        # update coordinates
        coords_updated = coords_updated + forces

        # count number of assigned spots
        x = np.around(coords_updated).astype(int)
        perc = np.sum(masks_binary[x[:, 0], x[:, 1], x[:, 2]]) / coords_updated.shape[0]
        print(f'iteration: {iii}    percent spots assigned: {perc}')

        # run callback function
        if callback is not None: callback(**locals())

    # determine final assignments and number of spots per segment
    coords_updated = np.around(coords_updated).astype(int)
    assignments = masks[coords_updated[:, 0], coords_updated[:, 1], coords_updated[:, 2]]
    counts = {x:0 for x in np.unique(masks)}
    for assignment in assignments: counts[assignment] += 1

    # return
    return counts, assignments


def gravity_flow_old(
    spots,
    masks,
    spacing,
    iterations,
    sigma,
    learning_rate,
    max_step,
    max_displacement,
    mask_density,
    callback=None,
):
    """
    Assign coordinate data to segment data by flowing coordinates along gravity gradients

    Parameters
    ----------
    spots : 2d array, Nxd for N spots in d dimensions
        The spots we want to assign

    masks : nd-array
        The multi-integer map of masks we want to assign spots to

    spacing : 1d-array
        The voxel spacing of the masks image

    iterations : int
        The number of iterations to run the flow

    sigma : float
        The standard deviation of the Gaussian applied to spots to form potential field

    learning_rate :

    max_step :

    max_displacement :

    mask_density :

    callback : function (default: None)

    Returns
    -------
    
    """

    masks_binary = masks > 0
    spots_updated = np.copy(spots[:, :3])
    for iii in range(iterations):

        # get spots as voxel coordinates
        coords = np.round( spots_updated / spacing ).astype(int)
        coords[ coords < 1 ] = 1
        coords[:, 0][coords[:, 0] > masks.shape[0] - 2] = masks.shape[0] - 2
        coords[:, 1][coords[:, 1] > masks.shape[1] - 2] = masks.shape[1] - 2
        coords[:, 2][coords[:, 2] > masks.shape[2] - 2] = masks.shape[2] - 2

        # get density
        spot_image = mask_density * masks_binary
        for coord in coords: spot_image[coord[0], coord[1], coord[2]] += 1
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(12)
        sitk_image = sitk.GetImageFromArray(spot_image.astype(np.float32))
        sitk_image.SetSpacing(spacing[::-1])
        sitk_image = sitk.SmoothingRecursiveGaussian(sitk_image, sigma=sigma)
        density = sitk.GetArrayFromImage(sitk_image)
    
        crops = np.empty((len(coords), 3, 3, 3), dtype=density.dtype)
        for jjj, coord in enumerate(coords):
            crops[jjj] = density[coord[0]-1:coord[0]+2,
                                 coord[1]-1:coord[1]+2,
                                 coord[2]-1:coord[2]+2]

        # get density gradient at spot locations, apply max_step
        grad = np.array(np.gradient(crops, *spacing, axis=(1, 2, 3))).transpose(1,2,3,4,0)
        grad = grad[:, 1, 1, 1, :].squeeze()
        grad_mag = np.linalg.norm(grad, axis=-1)
        scale_factor = learning_rate / np.mean(grad_mag)
        grad *= scale_factor
        grad_mag *= scale_factor  # scaling a vector equivalently scales its magnitude
        too_big = grad_mag > max_step
        grad[too_big] = grad[too_big] * max_step / grad_mag[too_big][..., None]

        # don't move anything too far
        displacements = spots_updated + grad
        too_far = np.linalg.norm(displacements - spots[:, :3], axis=-1) > max_displacement
        grad[too_far] = 0
        print(f'iteration: {iii}    gradient size: {np.sum(grad**2)}')
        spots_updated += grad

        # run callback function
        if callback is not None: callback(**locals())

    # get spots as voxel coordinates
    coords = np.round( spots_updated / spacing ).astype(int)
    coords[ coords < 1 ] = 1
    coords[:, 0][coords[:, 0] > masks.shape[0] - 2] = masks.shape[0] - 2
    coords[:, 1][coords[:, 1] > masks.shape[1] - 2] = masks.shape[1] - 2
    coords[:, 2][coords[:, 2] > masks.shape[2] - 2] = masks.shape[2] - 2

    assignments = masks[coords[:, 0], coords[:, 1], coords[:, 2]]
    counts = np.zeros(masks.max(), dtype=np.uint16)
    for assignment in assignments:
        if assignment > 0: counts[assignment - 1] += 1

    # TODO: add something which introspects the unassigned spot locations
    return counts, assignments


def simulate_and_integrate(
    spots,
    masks,
    spacing,
    sigma,
):
    """
    """

    # get spots as voxel coordinates
    coords = np.round( spots[:, :3] / spacing ).astype(int)
    coords[ coords < 1 ] = 1
    coords[:, 0][coords[:, 0] > masks.shape[0] - 2] = masks.shape[0] - 2
    coords[:, 1][coords[:, 1] > masks.shape[1] - 2] = masks.shape[1] - 2
    coords[:, 2][coords[:, 2] > masks.shape[2] - 2] = masks.shape[2] - 2

    # get intensities as percentiles
    percentiles = np.arange(101, dtype=np.float32)
    intensities = np.percentile(spots[:, -1], percentiles)
    remapped = np.interp(spots[:, -1], intensities, percentiles)

    # get density
    spot_image = np.zeros(masks.shape, dtype=remapped.dtype)
    for coord, intensity in zip(coords, remapped):
        spot_image[coord[0], coord[1], coord[2]] += intensity
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(12)
    sitk_image = sitk.GetImageFromArray(spot_image)
    sitk_image.SetSpacing(spacing[::-1])
    sitk_image = sitk.SmoothingRecursiveGaussian(sitk_image, sigma=sigma)
    density = sitk.GetArrayFromImage(sitk_image)

    # integrate density within cells
    signal = []
    boxes = find_objects(masks)
    for iii, box in enumerate(boxes):
        if box is not None:
            a = density[box]
            b = masks[box]
            signal.append(np.sum(a[b == iii]))
        else:
            signal.append(0)
    return density, signal

