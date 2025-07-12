import numpy as np
import pytest
from spex import (
    load_image,
    median_denoise,
    stardist_cellseg,
    nlm_denoise,
    watershed_classic,
    background_subtract,
    cellpose_cellseg,
    rescue_cells,
    simulate_cell,
    remove_small_objects,
    remove_large_objects,
    feature_extraction_adata,
)
import cv2
from skimage.draw import disk
from aicsimageio.writers import OmeTiffWriter
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import label, regionprops


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


def test_nlm_denoise_accuracy(test_ome_path):
    array, channels = load_image(test_ome_path)
    channel = array[0]

    sigma_est = np.mean(estimate_sigma(channel, channel_axis=None))
    expected = denoise_nl_means(
        channel,
        h=0.6 * sigma_est,
        sigma=sigma_est,
        fast_mode=True,
        patch_size=3,
        patch_distance=4,
        channel_axis=None,
        preserve_range=True,
    )

    denoised = nlm_denoise(array, patch=3, dist=4)

    np.testing.assert_allclose(denoised[0], expected, rtol=1e-5, atol=1e-8)
    assert np.any(denoised[0] > 0)
    assert np.any(expected > 0)


def test_watershed_classic(test_ome_path):
    array, channels = load_image(test_ome_path)
    labels = watershed_classic(array, [0])
    assert labels.shape == array.shape[1:]
    assert np.max(labels) > 0


def test_background_subtract_basic():
    # Create a synthetic 3-channel image of shape (C, X, Y)
    image = np.zeros((3, 10, 10), dtype=np.float32)
    image[0] += 50   # background channel
    image[1] += 100  # channel 1
    image[2] += 150  # channel 2

    # Add a bright spot to simulate signal in background
    image[0, 5, 5] = 255

    # Apply background subtraction
    result = background_subtract(image.copy(), channel=0, threshold=200, subtraction=30)

    # Shape must be unchanged
    assert result.shape == image.shape

    # Subtracted result should not be greater than original
    assert np.all(result[1] <= image[1])
    assert np.all(result[2] <= image[2])

    # Result should contain no negative values
    assert np.all(result >= 0)


def test_cellpose_centers_and_diameter_sensitivity():
    img = np.zeros((1, 256, 256), dtype=np.uint8)
    cv2.circle(img[0], (50, 64), 20, 255, -1)
    cv2.circle(img[0], (120, 64), 20, 255, -1)
    cv2.circle(img[0], (50, 150), 20, 255, -1)
    cv2.circle(img[0], (78, 150), 20, 255, -1)
    cv2.circle(img[0], (150, 200), 20, 255, -1)
    cv2.circle(img[0], (152, 200), 20, 255, -1)

    array = img.astype(np.float32)
    centers = [(50, 64), (120, 64), (50, 150), (78, 150), (150, 200), (152, 200)]

    def count_detected(array, centers, diameter):
        labels = cellpose_cellseg(array, [0], diameter=diameter, scaling=1)
        detected = set()
        for x, y in centers:
            label = labels[y, x]
            if label > 0:
                detected.add(label)
        return len(detected), labels

    d40_count, _ = count_detected(array, centers, diameter=40)
    d20_count, _ = count_detected(array, centers, diameter=20)
    d10_count, _ = count_detected(array, centers, diameter=10)

    assert d40_count >= d20_count >= d10_count, (
        f"Expected decreasing detection: d40={d40_count}, d20={d20_count}, d10={d10_count}"
    )
    assert d40_count >= 3, "Too few cells detected with diameter=40"


def test_rescue_cells_adds_missing_objects():
    from skimage.draw import disk
    from skimage.measure import label, regionprops

    # Create synthetic image (2 channels, 128x128)
    img = np.zeros((2, 128, 128), dtype=np.float32)

    # Channel 0: bright nucleus
    rr, cc = disk((40, 64), 12)
    img[0, rr, cc] = 0.9

    # Channel 1: second nucleus
    rr2, cc2 = disk((90, 64), 12)
    img[1, rr2, cc2] = 0.6

    # Simulate initial segmentation (misses 2nd nucleus)
    seg = np.zeros((128, 128), dtype=np.int32)
    seg[img[0] > 0.9] = 1
    seg = label(seg)

    # Apply rescue
    combined = rescue_cells(img, seg_channels=[0, 1], label_ling=seg)

    # Count large objects only
    props = regionprops(combined)
    count = sum(1 for p in props if p.area >= 100)
    print(f"Areas: {[p.area for p in props]}")

    assert count == 2, f"Expected 2 cells after rescue, got {count}"


def test_simulate_cell():
    from skimage.draw import disk
    from skimage.measure import label

    # Create an empty image with a single labeled object
    labels = np.zeros((128, 128), dtype=np.int32)
    rr, cc = disk((64, 64), 10)
    labels[rr, cc] = 1

    # Simulate dilation
    dilated = simulate_cell(labels, dist=5)

    assert dilated.shape == labels.shape
    assert np.max(dilated) >= np.max(labels)
    assert np.sum(dilated > 0) > np.sum(labels > 0)


def test_remove_small_and_large_objects_detailed():
    from skimage.draw import disk
    from skimage.measure import label, regionprops

    image = np.zeros((256, 256), dtype=np.uint8)

    # 3 small
    rr1, cc1 = disk((20, 20), 3)
    rr2, cc2 = disk((20, 40), 4)
    rr3, cc3 = disk((20, 60), 5)
    image[rr1, cc1] = 1
    image[rr2, cc2] = 1
    image[rr3, cc3] = 1

    # 1 nornal
    rr4, cc4 = disk((128, 128), 12)
    image[rr4, cc4] = 1

    # 2 big
    rr5, cc5 = disk((200, 60), 25)
    rr6, cc6 = disk((200, 160), 30)
    image[rr5, cc5] = 1
    image[rr6, cc6] = 1

    labels = label(image)
    original_props = regionprops(labels)
    original_areas = sorted([p.area for p in original_props])

    # check how many
    assert len(original_areas) == 6, f"Expected 6 initial objects, got {len(original_areas)}"

    # remove small
    cleaned_small = remove_small_objects(labels, minsize=100)
    small_removed_labels = label(cleaned_small)
    props_after_small = regionprops(small_removed_labels)
    areas_after_small = sorted([p.area for p in props_after_small])

    assert len(areas_after_small) == 3, f"Expected 3 objects after small removal, got {len(areas_after_small)}"
    assert all(a >= 100 for a in areas_after_small), f"Found small object(s) still present: {areas_after_small}"

    # remove big
    cleaned_both = remove_large_objects(cleaned_small, maxsize=1500)
    final_labels = label(cleaned_both)
    final_props = regionprops(final_labels)
    final_areas = [p.area for p in final_props]

    assert len(final_areas) == 1, f"Expected 1 object after large removal, got {len(final_areas)}"
    assert 100 <= final_areas[0] <= 1500, f"Remaining object size out of expected range: {final_areas[0]}"


def test_feature_extraction_various_cases():
    from skimage.draw import disk
    import numpy as np

    # Case 1: Multiple labeled cells
    img = np.zeros((3, 64, 64), dtype=np.float32)
    labels = np.zeros((64, 64), dtype=np.int32)

    rr1, cc1 = disk((20, 20), 5)
    rr2, cc2 = disk((45, 45), 7)

    labels[rr1, cc1] = 1
    labels[rr2, cc2] = 2

    img[0, rr1, cc1] = 100
    img[1, rr2, cc2] = 150
    img[2, rr2, cc2] = 200

    channels = ["ch0", "ch1", "ch2"]
    adata = feature_extraction_adata(img, labels, channels)

    assert adata.shape[0] == 2
    assert set(adata.var_names) == set(channels)
    assert "x_coordinate" in adata.obs.columns
    assert "cell_polygon" in adata.obsm
    assert adata.layers["X_uint8"].shape == adata.X.shape

    # Case 2: One tiny labeled cell
    edge_labels = np.zeros((64, 64), dtype=np.int32)
    rr3, cc3 = disk((10, 10), 2)
    edge_labels[rr3, cc3] = 1
    edge_img = np.random.rand(3, 64, 64).astype(np.float32)
    edge_img[:, rr3, cc3] += 1.0  # make sure signal is high

    adata_edge = feature_extraction_adata(edge_img, edge_labels, channels)

    assert adata_edge.shape[0] == 1
    assert set(adata_edge.var_names) == set(channels)
    assert "x_coordinate" in adata_edge.obs.columns
    assert "cell_polygon" in adata_edge.obsm
    assert adata_edge.layers["X_uint8"].shape == adata_edge.X.shape