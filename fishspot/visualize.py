import numpy as np


def spot_image(reference, spots, spacing, radius):
    """
    """

    spot_img = np.zeros_like(reference)
    coords = (spots[:, :3] / spacing).astype(int)
    r = radius  # shorthand
    for coord in coords:
        slc = tuple(slice(x-r, x+r) for x in coord)
        spot_img[slc] = 1
    return spot_img

