# fishspot
Tools for finding discrete bright spots in images.
* Background subtraction
* Fully automated PSF estimation
* Deconvoluion
* Blob detection
* Distributed pipeline for massive 3D images

|                          |                          |
:-------------------------:|:-------------------------:
![spots1](resources/full_res_spots_example.gif)  |  ![spots2](resources/full_res_spots_example2.gif)

Fishspot estimates a coordinate for each discrete bright spot in an image. Built for large 3D data, Fishspot includes a pipeline for distributing computations on overlapping blocks. The fully data driven automated PSF estimation may be useful outside of the spot detection context as well.

## Installation
`pip install fishspot`

## Documentation
I have not had time to complete docstrings, a lot of work is needed here. However, many of the functions are simple and a quick look at the source will explain them. In the meantime, **please use the github issue tracker for questions and support.**

## Modules
`fishspot.filter`
* background subtraction
* deconvolution

`fishspot.psf`
* automated psf estimation from point source data

`fishspot.detect`
* blob detection, typically run after filtering

`fishspot.distribute`
* distributed spot detection pipeline for massive 3D images
* a good example of how to combine the previous modules into a working pipeline

`fishspot.filter_spots`
* a few tools for filtering coordinates after they're found

## Dependencies
For filtering and detection, fishspot basically wraps some already excellent functions. The automated psf estimation may not exist elsewhere, though it's basic concept (RANSAC) certainly does. Either way, these excellent libraries make this possible:
* [scikit-image](https://github.com/scikit-image/scikit-image)
* [numpy](https://github.com/numpy/numpy)
* [dask](https://github.com/dask/dask)
* [ClusterWrap](https://github.com/GFleishman/ClusterWrap)
