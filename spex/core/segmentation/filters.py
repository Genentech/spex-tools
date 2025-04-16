import numpy as np
from skimage.filters import median
from skimage.morphology import disk
from skimage.util import apply_parallel


# def convert_mibitiff2zarr(inputpath, outputpath):
#     """convert mibi tiff to zarr

#     Parameters
#     ----------
#     inputpath : Path of tiff file
#     outputpath : Path of ometiff or omezarr file. Note: for omezarr, the path must end in *.zarr/0

#     """
#     img = AICSImage(inputpath)
#     im_array = img.get_image_data("ZYX", T=0, C=0)

#     Channel_list = []
#     with TiffFile(inputpath) as tif:
#         for page in tif.pages:
#             # get tags as json
#             description = json.loads(page.tags['ImageDescription'].value)
#             Channel_list.append(description['channel.target'])

#     writer = OmeZarrWriter(outputpath)
#     writer.write_image(im_array, image_name="Image:0", dimension_order="CYX", channel_names=Channel_list,
#                        scale_num_levels=4, physical_pixel_sizes=None, channel_colors=None)

#     print('conversion complete')






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