from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter, OmeZarrWriter
from aicsimageio.readers import BioformatsReader
from tifffile import imread, imsave, TiffWriter, imwrite, TiffFile
import json


def load_image(imgpath: str = "") -> tuple:
    """Load image and check/correct for dimension ordering.

    This function loads multi-channel microscopy images in OME-TIFF or OME-ZARR format
    and automatically handles dimension ordering to ensure proper channel separation.
    It supports various image formats through BioformatsReader and extracts channel
    information from image metadata.

    Parameters
    ----------
    imgpath : str
        Path to the image file. Supports OME-TIFF (.ome.tiff, .ome.tif) and
        OME-ZARR (.zarr) formats. For OME-ZARR files, the path must end with
        `*.zarr/0` to specify the correct subdirectory.

    Returns
    -------
    tuple
        A tuple containing:
        - array : numpy.ndarray
            Multi-channel image data with shape (C, Y, X) where C is the number
            of channels, Y and X are spatial dimensions
        - channel_list : list
            List of channel names extracted from image metadata. If metadata
            is not available, generic channel names may be used.

    Examples
    --------
    Load an OME-TIFF file:

    >>> from spex import load_image
    >>> array, channels = load_image("path/to/image.ome.tiff")
    >>> print(f"Image shape: {array.shape}")
    >>> print(f"Channels: {channels}")

    Load an OME-ZARR file:

    >>> array, channels = load_image("path/to/image.zarr/0")
    >>> print(f"Number of channels: {array.shape[0]}")

    Notes
    -----
    - The function automatically detects the channel dimension and reorders
      the data to ensure channels are in the first dimension
    - Channel names are extracted from OME metadata when available
    - For TIFF files without proper metadata, channel names may be empty
      or generic
    - The function uses BioformatsReader for maximum format compatibility
    """

    img = AICSImage(imgpath, reader=BioformatsReader)

    dims = ["T", "C", "Z"]
    shape = list(img.shape)
    channel_dim = dims[shape.index(max(shape[0:3]))]

    array = img.get_image_data(channel_dim + "YX")

    channel_list = img.channel_names

    if len(channel_list) != array.shape[0]:
        channel_list = []
        with TiffFile(imgpath) as tif:
            for page in tif.pages:
                # get tags as json
                description = json.loads(page.tags["ImageDescription"].value)
                channel_list.append(description["channel.target"])
    channel_list = [ch.replace("Target:", "") if ch else ch for ch in channel_list]

    return array, channel_list
