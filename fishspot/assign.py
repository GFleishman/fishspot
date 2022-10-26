import numpy as np
import SimpleITK as sitk
from scipy.ndimage import find_objects


def gravity_flow(
    spots,
    masks,
    spacing,
    iterations,
    sigma,
    learning_rate,
    max_step,
    max_displacement,
    mask_density,
    free_iteration_percentage=0.1,
    callback=None,
):
    """
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
    
        # cut out only regions around spots, ignore trapped spots
        if iii < round( iterations * free_iteration_percentage ):
            trapped = np.zeros(coords.shape[0], dtype=bool)
        else:
            trapped = masks_binary[coords[:, 0], coords[:, 1], coords[:, 2]].astype(bool)
        coords = coords[~trapped]
        crops = np.empty((len(coords), 3, 3, 3), dtype=density.dtype)
        for jjj, coord in enumerate(coords):
            crops[jjj] = density[coord[0]-1:coord[0]+2,
                                 coord[1]-1:coord[1]+2,
                                 coord[2]-1:coord[2]+2]
            
        # print some feedback
        print(iii, ": ", np.sum(trapped) / len(trapped))

        # get density gradient at spot locations, apply max_step
        grad = np.array(np.gradient(crops, *spacing, axis=(1, 2, 3))).transpose(1,2,3,4,0)
        grad = grad[:, 1, 1, 1, :].squeeze()
        grad_mag = np.linalg.norm(grad, axis=-1)
        grad *= learning_rate / np.mean(grad_mag)
        too_big = grad_mag > max_step
        grad[too_big] = grad[too_big] * max_step / grad_mag[too_big][..., None]
    
        # don't move anything too far
        displacements = spots_updated[~trapped] + grad
        too_far = np.linalg.norm(displacements - spots[:, :3][~trapped], axis=-1) > max_displacement
        grad[too_far] = 0
        spots_updated[~trapped] -= grad

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
    for assignment in assignments: counts[assignment] += 1
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

