import numpy as np
import os
import re
import urllib.request
from urllib.parse import urljoin


def to_uint8(arr, norm_along=None):
    """
    Convert an array to uint8 format with normalization.

    Parameters
    ----------
    arr : np.ndarray
        The input array to normalize and convert.
    norm_along : str or None
        Direction of normalization:
        - "global": normalize over the entire array.
        - "var": normalize over columns (variables).
        - "obs": normalize over rows (observations).
        - None: defaults to "global".

    Returns
    -------
    np.ndarray
        The normalized and scaled array as uint8.
    """
    arr = np.asarray(arr)

    if norm_along == "var":
        # Normalize each column independently
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # Prevent division by zero
        scaled = (arr - min_vals) / ranges

    elif norm_along == "obs":
        # Normalize each row independently
        min_vals = arr.min(axis=1)[:, np.newaxis]
        max_vals = arr.max(axis=1)[:, np.newaxis]
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        scaled = (arr - min_vals) / ranges

    else:
        # Global normalization
        min_val = arr.min()
        max_val = arr.max()
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1
        scaled = (arr - min_val) / range_val

    return (scaled * 255).astype(np.uint8)


def download_cellpose_models():
    MODEL_DIR = os.path.expanduser("~/.cellpose/models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    BASE_URL = "http://65.108.226.226:8080/cellpose_models/"

    try:
        with urllib.request.urlopen(BASE_URL) as response:
            html = response.read().decode()
    except Exception as e:
        print(f"❌ Cannot reach model server: {e}")
        return

    files = re.findall(r'href="([^"/][^"]+)"', html)
    print(f"📦 {len(files)} models found")

    for name in files:
        dest = os.path.join(MODEL_DIR, name)
        url = urljoin(BASE_URL, name)

        if os.path.exists(dest):
            print(f"✔ {name} exists")
            continue

        print(f"🔽 Downloading {name}...", end=" ")
        try:
            urllib.request.urlretrieve(url, dest)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")