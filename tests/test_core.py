import os
import io
import re
import sys
import time
import socket
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urljoin
import hashlib

import pytest


REMOTE_BASE = "http://65.108.226.226:8080/cellpose_models/"


class RangeCaptureHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves files from a given directory and logs Range requests."""
    range_log = []  # list of tuples (path, range_header)

    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        rng = self.headers.get("Range")
        self.__class__.range_log.append((self.path, rng))
        return super().do_GET()

    def log_message(self, format, *args):
        # silence HTTP server logs to keep test output clean
        pass


class _FakeResponse:
    """Minimal stub for urlopen(BASE) -> returns an HTML page with links."""
    def __init__(self, data: bytes):
        self._fp = io.BytesIO(data)

    def read(self, amt=-1):
        return self._fp.read(amt)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._fp.close()

    @property
    def headers(self):
        # not used for the HTML page
        return {}


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Override HOME so download_cellpose_models writes to ~/.cellpose/models under tmp."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def http_server(tmp_path):
    """Start a local HTTP server with Range support; yield (base_url, directory, server)."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()

    handler = partial(RangeCaptureHandler, directory=str(serve_dir))
    # port 0 -> OS will choose a free port
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/"

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    yield base_url, serve_dir, server

    server.shutdown()
    server.server_close()


def _write_file(path, size_bytes, pattern=True):
    """Create a file of the given size; deterministic content for hash checks."""
    with open(path, "wb") as f:
        if pattern:
            # repeat bytes 0..255
            block = bytes([i % 256 for i in range(8192)])
            left = size_bytes
            while left > 0:
                n = min(left, len(block))
                f.write(block[:n])
                left -= n
        else:
            f.write(os.urandom(size_bytes))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_download_cellpose_models_multipart_resume_cleanup(tmp_home, http_server, monkeypatch, capsys):
    # Prepare files on the local server
    local_base, serve_dir, server = http_server

    big_src = serve_dir / "big.bin"
    small_src = serve_dir / "small.bin"

    # > 5 MiB to trigger multipart (threshold in downloader is 5*1024*1024)
    _write_file(big_src, size_bytes=8 * 1024 * 1024 + 123, pattern=True)
    _write_file(small_src, size_bytes=3 * 1024, pattern=True)

    big_hash = _sha256(big_src)
    small_hash = _sha256(small_src)

    # HTML page that the "remote" BASE_URL will return
    html = f"""
    <html>
      <body>
        <a href="big.bin">big.bin</a>
        <a href="small.bin">small.bin</a>
      </body>
    </html>
    """.encode()

    real_urlopen = urllib.request.urlopen

    def fake_urlopen(req, *args, **kwargs):
        # Extract URL from Request or string
        if isinstance(req, urllib.request.Request):
            url = req.full_url
            headers = req.headers
        else:
            url = req
            headers = {}

        # Substitute only for BASE_URL (return HTML with links)
        if url == REMOTE_BASE:
            return _FakeResponse(html)

        # Rewrite model file URLs to the local server, preserving Range headers
        if url == urljoin(REMOTE_BASE, "big.bin"):
            new_req = urllib.request.Request(urljoin(local_base, "big.bin"), headers=headers)
            return real_urlopen(new_req, *args, **kwargs)

        if url == urljoin(REMOTE_BASE, "small.bin"):
            new_req = urllib.request.Request(urljoin(local_base, "small.bin"), headers=headers)
            return real_urlopen(new_req, *args, **kwargs)

        # Fallback: default behavior
        return real_urlopen(req, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen, raising=True)

    # Import the function under test
    from spex.core.utils import download_cellpose_models

    # Create stale .part files beforehand to verify cleanup on start
    model_dir = tmp_home / ".cellpose" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    stale_part = model_dir / "big.bin.part07"
    stale_part.write_bytes(b"stale")

    # Mock input to automatically answer "y"
    def mock_input(prompt):
        return "y"
    
    monkeypatch.setattr("builtins.input", mock_input)
    
    # Run the download
    download_cellpose_models()

    captured = capsys.readouterr().out

    # Assertions:
    # 1) Files are downloaded
    big_dst = model_dir / "big.bin"
    small_dst = model_dir / "small.bin"
    assert big_dst.exists(), "big.bin was not downloaded"
    assert small_dst.exists(), "small.bin was not downloaded"

    # 2) Hashes match source files
    assert _sha256(big_dst) == big_hash, "big.bin content is corrupted"
    assert _sha256(small_dst) == small_hash, "small.bin content is corrupted"

    # 3) Temporary .part* files are cleaned up
    assert not any(model_dir.glob("big.bin.part*")), "Residual .part files for big.bin were not removed"
    assert not any(model_dir.glob("small.bin.part*")), "Residual .part files for small.bin were not removed"
    assert not stale_part.exists(), "Pre-existing stale .part file was not removed"

    # 4) Progress output was printed for each file
    assert "big.bin" in captured and "✅ big.bin" in captured, "No progress/final line for big.bin"
    assert "small.bin" in captured and "✅ small.bin" in captured, "No progress/final line for small.bin"

    # 5) Range requests were used for the big file (multipart)
    ranges_for_big = [rng for path, rng in RangeCaptureHandler.range_log if path.endswith("/big.bin")]
    assert any(rng for rng in ranges_for_big), "Range was not used for big.bin (multipart expected)"

    # 6) Range was not used for the small file (single-part)
    ranges_for_small = [rng for path, rng in RangeCaptureHandler.range_log if path.endswith("/small.bin")]
    assert all(rng is None for rng in ranges_for_small), "Range unexpectedly used for small.bin"