import numpy as np
from skimage.filters import median
from skimage.morphology import disk
from skimage.util import apply_parallel
from skimage.restoration import denoise_nl_means, estimate_sigma


def median_denoise(image: np.ndarray, kernel: int, ch: list[int]) -> np.ndarray:
    """
    Median denoising for selected channels of a multichannel image.

    Parameters
    ----------
    image : np.ndarray
        Multichannel image array with shape (C, X, Y).
    kernel : int
        Kernel size for the median filter (typical range: 5–7).
    ch : list[int]
        List of channel indices to denoise.

    Returns
    -------
    np.ndarray
        Denoised image stack with shape (C, X, Y).
    """
    def median_denoise_wrap(array):
        correct = array[0]
        correct = median(correct, disk(kernel))
        return correct[np.newaxis, ...]

    denoise = apply_parallel(
        median_denoise_wrap,
        image,
        chunks=(1, image.shape[1], image.shape[2]),
        dtype="float",
        compute=True,
    )

    filtered = []
    for i in range(image.shape[0]):
        temp = denoise[i] if i in ch else image[i]
        temp = np.expand_dims(temp, 0)
        filtered.append(temp)

    return np.concatenate(filtered, axis=0)


def nlm_denoise(image, patch:int = 5, dist:int = 6) -> np.ndarray:

    """Non local means denoising

    Parameters
    ----------
    image : Multichannel numpy array (C,X,Y)
    patch: int, Patch size (5 is typical)
    dist: int, ignore pixels above this threshold (6 is typical)

    Returns
    -------
    Image Stack : Denoised image stack as numpy array (C,X,Y)

    """

    def nlm_denoise_wrap(array):
        correct = array[0]
        sigma_est = np.mean(estimate_sigma(correct, channel_axis=None))
        correct = denoise_nl_means(
            correct,
            h=0.6 * sigma_est,
            sigma=sigma_est,
            fast_mode=True,
            patch_size=patch,
            patch_distance=dist,
            channel_axis=None,
            preserve_range=True,
        )
        return correct[np.newaxis, ...]

    denoise = apply_parallel(
        nlm_denoise_wrap,
        image,
        chunks=(1, image.shape[1], image.shape[2]),
        dtype="float",
        compute=True,
    )

    return denoise