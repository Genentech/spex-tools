import numpy as np
import pytest
from spex import say_hello, load_image, median_denoise, stardist_cellseg
from aicsimageio.writers import OmeTiffWriter

@pytest.fixture(scope="module")
def test_ome_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test_image.ome.tiff"
    data = np.random.randint(0, 256, size=(1, 2, 1, 64, 64), dtype=np.uint16)  # T, C, Z, Y, X

    OmeTiffWriter.save(
        data,
        path,
        dim_order="TCZYX",
        channel_names=["DAPI", "CD45"]
    )

    return str(path)


def test_say_hello():
    assert say_hello("Test") == "Hello, Test!"

def test_load_file(test_ome_path):
    array, channels = load_image(test_ome_path)
    assert array.shape[0] == len(channels)
    assert channels == ["DAPI", "CD45"]

def test_median_denoise(test_ome_path):
    array, channels = load_image(test_ome_path)
    denoised_array = median_denoise(array, 5, [0, 1])
    
    assert denoised_array.shape == array.shape
    assert np.any(denoised_array != array)  

def test_stardist(test_ome_path):
    array, channels = load_image(test_ome_path)
    scaling = 1
    threshold = 0.479071
    _min = float(1)
    _max = float(98.5)

    labels = stardist_cellseg(
        array,
        [0, 1],  
        scaling,
        threshold,
        _min,
        _max
    )
    
    assert labels.shape == array.shape[1:]  # Check if labels have the same spatial dimensions as the input image
    assert np.max(labels) > 0  # Check if some labels were generated