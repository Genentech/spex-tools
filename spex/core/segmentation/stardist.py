import numpy as np
from stardist.models import StarDist2D
import cv2
from csbdeep.utils import normalize
import re

def stardist_cellseg(image, seg_channels, scaling, threshold, _min, _max):
    """
    Segment image by stardist deeplearning method

    Parameters
    ----------
    image : Multichannel image as numpy array
    seg_channels: list of indices to use for nuclear segmentation
    scaling: Integer value scaling
    threshold: probability cutoff
    _min: bottom percentile normalization
    _max: top percentile normalization

    Returns
    -------
    labels : per cell segmentation as numpy array
    """
    temp2 = np.zeros((image.shape[1], image.shape[2]))
    for i in seg_channels:
        try:
            temp = image[i]
            temp2 += temp
        except IndexError:
            print(f"Index {i} out of bounds for image with shape {image.shape}")

    seg_image = cv2.resize(
        temp2,
        (temp2.shape[1] * scaling, temp2.shape[0] * scaling),
        interpolation=cv2.INTER_NEAREST,
    )

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    image_norm = normalize(seg_image, _min, _max)
    labels, _ = model.predict_instances(image_norm, prob_thresh=threshold)

    labels = cv2.resize(
        labels,
        (labels.shape[1] // scaling, labels.shape[0] // scaling),
        interpolation=cv2.INTER_NEAREST,
    )

    return labels

def run(**kwargs):
    image = kwargs.get('image')
    channel_list = kwargs.get("channel_list", [])
    all_channels = kwargs.get("all_channels", [])

    all_channels = [re.sub("[^0-9a-zA-Z]", "", item).lower().replace("target", "") for item in all_channels]
    channel_list = [re.sub("[^0-9a-zA-Z]", "", item).lower().replace("target", "") for item in channel_list]

    channel_list = [
        all_channels.index(channel)
        for channel in channel_list
        if channel in all_channels
    ]
    channel_list.sort()

    scaling = int(kwargs.get('scaling', 1))
    threshold = float(kwargs.get('threshold', 0.479071))
    _min = float(kwargs.get('_min', 1))
    _max = float(kwargs.get('_max', 98.5))

    labels = stardist_cellseg(
        image,
        channel_list,
        scaling,
        threshold,
        _min,
        _max
    )

    return {
        'labels': labels
    }
