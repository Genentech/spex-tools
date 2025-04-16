from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter, OmeZarrWriter
from tifffile import imread, imsave, TiffWriter, imwrite, TiffFile
import json

def load_image(imgpath: str = '') -> str:
    """Load image and check/correct for dimension ordering

    Parameters
    ----------
    img : Path of ometiff or omezarr file. Note: for omezarr, the path must end in *.zarr/0

    Returns
    -------
    Image Stack : 2D numpy array
    Channels : list

    """

    img = AICSImage(imgpath)

    dims = ['T', 'C', 'Z']
    shape = list(img.shape)
    channel_dim = dims[shape.index(max(shape[0:3]))]

    array = img.get_image_data(channel_dim + "YX")

    channel_list = img.channel_names

    if len(channel_list) != array.shape[0]:
        channel_list = []
        with TiffFile(imgpath) as tif:
            for page in tif.pages:
                # get tags as json
                description = json.loads(page.tags['ImageDescription'].value)
                channel_list.append(description['channel.target'])

    return array, channel_list