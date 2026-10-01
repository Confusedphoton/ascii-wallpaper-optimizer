"""Exercise the viewer's image registration and real FLIP HTTP pipeline."""

import json
import threading
from contextlib import contextmanager
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from scripts.view_renders import RenderCatalog, make_server


@contextmanager
def running_viewer(tmp_path):
    catalog = RenderCatalog()
    for name, size, color in (
        ("source", (64, 40), "#446688"),
        ("render", (64, 40), "#668844"),
        ("different_size", (32, 40), "#446688"),
    ):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size, color).save(path)
        catalog.add_images([f"{name}={path}"], 67)
    server = make_server(catalog, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", catalog
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def post_flip(base, first, second, ppd=67):
    request = Request(
        base + "/api/flip",
        data=json.dumps({"a": first, "b": second, "ppd": ppd}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def test_registered_images_and_ui(tmp_path):
    with running_viewer(tmp_path) as (base, catalog):
        with urlopen(base + "/api/catalog") as response:
            scenes = json.load(response)["scenes"]
        entry = scenes[0]["entries"][0]
        assert (entry["width"], entry["height"]) == (64, 40)
        assert entry["id"] in catalog.assets
        with urlopen(base + entry["url"]) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert Image.open(BytesIO(response.read())).size == (64, 40)
        with urlopen(base + "/") as response:
            assert b"FLIP error" in response.read()
        for path in ("/asset/unknown", "/asset/../../pyproject.toml", "/unknown"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404


def test_flip_actual_error_map_and_cache(tmp_path):
    pytest.importorskip("flip_evaluator")
    with running_viewer(tmp_path) as (base, catalog):
        first, second, different_size = catalog.assets
        identical = post_flip(base, first, first)
        assert identical["mean_error"] == pytest.approx(0, abs=1e-7)
        result = post_flip(base, first, second)
        assert 0 < result["mean_error"] < 1
        assert result["cached"] is False
        cached = post_flip(base, first, second)
        assert cached["cached"] is True
        assert cached["url"] == result["url"]
        with urlopen(base + result["url"]) as response:
            assert Image.open(BytesIO(response.read())).size == (64, 40)
        for a, b, ppd in (
            (first, different_size, 67),
            (first, second, "nan"),
            (first, second, 0),
            (first, "unknown", 67),
        ):
            with pytest.raises(HTTPError) as error:
                post_flip(base, a, b, ppd)
            assert error.value.code == 400
