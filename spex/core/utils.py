import numpy as np
import os
import re
import sys
import time
import math
import glob
import threading
import urllib.request
from urllib.parse import urljoin
from urllib.error import URLError, HTTPError
import logging


def to_uint8(arr, norm_along=None):
    """
    Convert an array to uint8 with optional normalization along rows/cols/global.
    """
    arr = np.asarray(arr)

    if norm_along == "var":
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        scaled = (arr - min_vals) / ranges

    elif norm_along == "obs":
        min_vals = arr.min(axis=1)[:, np.newaxis]
        max_vals = arr.max(axis=1)[:, np.newaxis]
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        scaled = (arr - min_vals) / ranges

    else:
        min_val = arr.min()
        max_val = arr.max()
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1
        scaled = (arr - min_val) / range_val

    return (scaled * 255).astype(np.uint8)


# ====== Logger config: logs to stderr ======
_handler = logging.StreamHandler(stream=sys.stderr)
_handler.setLevel(logging.ERROR)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
logging.basicConfig(level=logging.ERROR, handlers=[_handler], force=True)
logger = logging.getLogger(__name__)

def _log_info(msg: str):
    logger.info(msg)
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

def _log_error(msg: str):
    logger.error(msg)
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

# ====== Downloader configuration ======
TIMEOUT = 30             # connect/read timeout (seconds)
RETRIES = 5              # retries per request
BUF = 1 << 20            # 1 MiB buffer
PARTS = 10               # parallel chunks for large files
MULTIPART_MIN = 5 << 20  # use multipart for files > 5 MiB

# ====== Internal helpers ======
def _range_probe(url):
    supports_range = False
    total = None
    try:
        req_head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req_head, timeout=TIMEOUT) as r:
            length = r.headers.get("Content-Length") or r.headers.get("content-length")
            if length and str(length).isdigit():
                total = int(length)
            accept = (r.headers.get("Accept-Ranges") or r.headers.get("accept-ranges") or "").lower()
            supports_range = "bytes" in accept
            return supports_range, total
    except Exception:
        pass

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            length = r.headers.get("Content-Length") or r.headers.get("content-length")
            if length and str(length).isdigit():
                total = int(length)
            accept = (r.headers.get("Accept-Ranges") or r.headers.get("accept-ranges") or "").lower()
            supports_range = supports_range or ("bytes" in accept)
            return supports_range, total
    except Exception:
        return supports_range, total

def _download_single(url, dest_part, start_byte=None, end_byte=None, progress_cb=None):
    attempt = 0
    while attempt < RETRIES:
        attempt += 1
        headers = {}
        if start_byte is not None:
            rng = f"bytes={start_byte}-" if end_byte is None else f"bytes={start_byte}-{end_byte}"
            headers["Range"] = rng
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dest_part, "wb") as f:
                while True:
                    chunk = r.read(BUF)
                    if not chunk:
                        break
                    f.write(chunk)
                    if progress_cb:
                        progress_cb(len(chunk))
            return True
        except (URLError, HTTPError, TimeoutError, ConnectionError):
            if attempt >= RETRIES:
                return False
            time.sleep(min(2 ** attempt, 30))
        except Exception:
            return False
    return False

def _merge_parts(part_files, final_dest):
    tmp = final_dest + ".tmp"
    with open(tmp, "wb") as out:
        for pf in part_files:
            with open(pf, "rb") as inp:
                while True:
                    chunk = inp.read(BUF)
                    if not chunk:
                        break
                    out.write(chunk)
    os.replace(tmp, final_dest)

def _human(n):
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.1f} {units[i]}"

# ====== SINGLE-LINE PROGRESS ======
def _progress_printer_single_line(name, total_size, get_downloaded, stop_event, poke_event, get_ok, interval=0.5):
    """
    Prints one progress line in stdout, overwriting it with '\r'.
    At the end prints '\n' and finishes only if successful; otherwise clears the line.
    """
    start = time.time()
    last_print = 0.0
    last_len = 0

    def _render(final=False):
        done = get_downloaded()
        elapsed = max(time.time() - start, 1e-6)
        speed = done / elapsed
        pct = (done / total_size * 100.0) if total_size else (100.0 if final else 0.0)
        prefix = "✅" if final else "🔽"
        line = f"{prefix} {name}: {pct:6.2f}%"
        nonlocal last_len
        pad = " " * max(0, last_len - len(line))
        sys.stdout.write("\r" + line + pad)
        if final:
            if get_ok():
                sys.stdout.write("\n")
            else:
                sys.stdout.write("\r" + " " * last_len + "\r")
        sys.stdout.flush()
        last_len = len(line)

    # draw something immediately (0%)
    _render(final=False)

    while not stop_event.is_set():
        poked = poke_event.wait(timeout=interval)
        if poked:
            poke_event.clear()
        now = time.time()
        if poked or (now - last_print) >= interval:
            _render(final=False)
            last_print = now

    _render(final=True)

def _download_file(url, dest):
    # Clean old parts
    for p in glob.glob(dest + ".part*"):
        try:
            os.remove(p)
        except OSError:
            pass

    supports_range, total = _range_probe(url)

    downloaded = 0
    lock = threading.Lock()
    poke_event = threading.Event()  # signal progress to update

    def add_progress(n):
        nonlocal downloaded
        with lock:
            downloaded += n
        poke_event.set()

    stop_event = threading.Event()

    finished_ok = {"ok": False}

    progress_thread = threading.Thread(
        target=_progress_printer_single_line,
        args=(os.path.basename(dest), total, lambda: downloaded, stop_event, poke_event, lambda: finished_ok["ok"], 0.5),
        daemon=True,
    )

    def _multipart_try():
        if not total or total < MULTIPART_MIN:
            return False
        part_size = math.ceil(total / PARTS)
        threads = []
        part_files = []
        errors = []

        def worker(part_path, start_b, end_b):
            ok = False
            attempt = 0
            while attempt < RETRIES and not ok:
                attempt += 1
                headers = {"Range": f"bytes={start_b}-{end_b}"}
                req = urllib.request.Request(url, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(part_path, "wb") as f:
                        status = getattr(r, "status", None) or r.getcode()
                        cr = r.headers.get("Content-Range") or r.headers.get("content-range")
                        if status != 206 or not cr:
                            raise RuntimeError("Server ignored Range")
                        while True:
                            chunk = r.read(BUF)
                            if not chunk:
                                break
                            f.write(chunk)
                            add_progress(len(chunk))
                    ok = True
                except Exception:
                    if attempt >= RETRIES:
                        errors.append(True)
                        return
                    time.sleep(min(2 ** attempt, 30))

        try:
            for i in range(PARTS):
                start_b = i * part_size
                end_b = min(total - 1, start_b + part_size - 1)
                if start_b > end_b:
                    continue
                pf = f"{dest}.part{i:02d}"
                part_files.append(pf)
                t = threading.Thread(target=worker, args=(pf, start_b, end_b), daemon=True)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            size_sum = sum(os.path.getsize(pf) for pf in part_files if os.path.exists(pf))
            if errors or size_sum != total:
                raise RuntimeError("Multipart size mismatch")

            _merge_parts(part_files, dest)
            for pf in part_files:
                try:
                    os.remove(pf)
                except OSError:
                    pass
            return True

        except Exception:
            for pf in part_files:
                try:
                    os.remove(pf)
                except OSError:
                    pass
            return False

    try:
        progress_thread.start()

        used_multipart = False
        if total and total >= MULTIPART_MIN:
            used_multipart = _multipart_try()

        if not used_multipart:
            part = dest + ".part"
            success = _download_single(url, part, None, None, add_progress)
            if not success:
                raise RuntimeError("single download failed")
            os.replace(part, dest)

        finished_ok["ok"] = True

        stop_event.set()
        progress_thread.join(timeout=2.0)
        return True

    except BaseException:
        finished_ok["ok"] = False
        stop_event.set()
        try:
            progress_thread.join(timeout=2.0)
        except Exception:
            pass
        return False

# ====== Public API ======
def download_cellpose_models():
    MODEL_DIR = os.path.expanduser("~/.cellpose/models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    BASE_URL = "http://65.108.226.226:8080/cellpose_models/"

    try:
        with urllib.request.urlopen(BASE_URL, timeout=TIMEOUT) as response:
            html = response.read().decode()
    except Exception:
        _log_error("❌ Cannot reach model server")
        return

    files = re.findall(r'href=["\\\']([^"\\\'<>?#/][^"\\\'<>?#]*)["\\\']', html)
    if not files:
        _log_error("Index has no parsable files")
        return

    missing = [name for name in files if not os.path.exists(os.path.join(MODEL_DIR, name))]
    if not missing:
        return

    answer = input(f"{len(missing)} models are missing. Download now? [y/N] (for cellpose): ").strip().lower()
    if answer != "y":
        return

    for name in missing:
        dest = os.path.join(MODEL_DIR, name)
        url = urljoin(BASE_URL, name)
        success = _download_file(url, dest)
        if not success:
            continue
    sys.stdout.write("\n")
    sys.stdout.flush()


def delete_cellpose_models():
    """
    Deletes the ~/.cellpose/models folder with all downloaded models.
    """
    import shutil
    model_dir = os.path.expanduser("~/.cellpose/models")
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
            print(f"🗑 Deleted models folder: {model_dir}")
        except Exception as e:
            print(f"❌ Failed to delete {model_dir}: {e}")
    else:
        print(f"ℹ Models folder not found: {model_dir}")