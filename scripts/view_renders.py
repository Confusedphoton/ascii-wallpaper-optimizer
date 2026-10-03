"""Local A/B render viewer with shared transforms and optional official FLIP maps.

  .venv/bin/python scripts/view_renders.py
  .venv/bin/python scripts/view_renders.py --report rendered/perceptual_experiments/pilot
  .venv/bin/python scripts/view_renders.py --image 'Source=/path/source.png' --image 'Render=/path/render.png'

The server only exposes registered images and its own UI, bound to localhost.
No renderer, wallpaper setting, or cached output is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

UI = Path(__file__).with_name("render_viewer")
LABELS = {
    "flow_static": "Flow · static baseline",
    "flow_animated": "Flow · animated",
    "flow_animated_smooth": "Flow · stronger temporal fitting",
    "flow_animated_stable": "Flow · maximum temporal fitting",
    "flow_animated_strong": "Flow · animated strong eddies",
    "perceptual_standard": "Perceptual · standard glyph bank",
    "perceptual_wide": "Perceptual · 96 glyphs",
    "flow_calm": "Flow domains · calm",
    "flow_default": "Flow domains · balanced",
    "flow_eddies": "Flow domains · stronger eddies",
    "flow_fine": "Flow domains · finer domains",
    "installed": "Installed render",
    "baseline": "Current baseline",
    "brightness_2": "Spatial brightness",
    "brightness_8": "Spatial brightness · stronger",
    "tensor": "Orientation tensor",
    "wasserstein": "Wasserstein orientation",
    "wasserstein_replace": "Wasserstein · Sobel replaced",
    "combined": "Brightness + Wasserstein · Sobel replaced",
    "combined_tensor": "Brightness + tensor · Sobel replaced",
    "combined_add": "Brightness + Wasserstein",
    "combined_tensor_add": "Brightness + tensor",
    "combined_strokes": "Brightness + stroke alignment",
}
STAGES = {
    "initial": "Initialization",
    "adam_step_6": "Adam · 6 updates",
    "adam_step_12": "Adam · 12 updates",
    "adam_step_24": "Adam · 24 updates",
    "solver_final": "Adam · terminal state",
}


class RenderCatalog:
    def __init__(self):
        self.assets = {}
        self.scenes = []

    def entry(self, path, label, kind="final"):
        path = Path(path).resolve(strict=True)
        with Image.open(path) as image:
            width, height = image.size
            if image.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError(f"Unsupported image format: {path.name}")
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:20]
        self.assets[identity] = path
        return {
            "id": identity,
            "label": label,
            "kind": kind,
            "url": f"/asset/{identity}",
            "width": width,
            "height": height,
        }

    def add_report(self, path):
        path = Path(path).resolve()
        report = json.loads((path / "metrics.json").read_text())
        for name, data in report["images"].items():
            directory = path / name
            entries = [self.entry(directory / "source.png", "Source", "source")]
            for variant in data["variants"]:
                label = LABELS.get(variant, variant.replace("_", " "))
                filename = directory / (
                    "installed_native.png"
                    if variant == "installed"
                    else f"{variant}/final.png"
                )
                if filename.exists():
                    entry = self.entry(filename, label)
                    clip = data["variants"][variant].get("clip")
                    if clip:
                        entry["frames"] = [self.entry(directory / variant / frame, f"{label} / frame {index}")
                                           for index, frame in enumerate(clip["frames"])]
                        entry["duration_ms"] = clip["duration_ms"]
                    entries.append(entry)
                for stage, title in STAGES.items():
                    filename = directory / variant / f"{stage}.png"
                    if filename.exists():
                        entries.append(
                            self.entry(filename, f"{label} / {title}", "intermediate")
                        )
            self.scenes.append(
                {
                    "id": name,
                    "label": data.get("label", "Vivian" if name == "viviana" else name.title()),
                    "ppd": data.get("ppd", 67),
                    "entries": entries,
                }
            )

    def add_images(self, images, ppd):
        entries = []
        for index, specification in enumerate(images):
            label, separator, path = specification.partition("=")
            if not separator:
                path, label = specification, Path(specification).stem
            entries.append(self.entry(path, label, "source" if index == 0 else "final"))
        self.scenes.append(
            {"id": "custom", "label": "Your images", "ppd": ppd, "entries": entries}
        )


class FlipMaps:
    def __init__(self, catalog, limit=8):
        self.catalog, self.limit = catalog, limit
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def compute(self, first, second, ppd):
        if first not in self.catalog.assets or second not in self.catalog.assets:
            raise ValueError("Choose images from the viewer catalog")
        ppd = float(ppd)
        if not math.isfinite(ppd) or not 1 <= ppd <= 1024:
            raise ValueError("Pixels per degree must be between 1 and 1024")
        identity = hashlib.sha256(f"{first}:{second}:{ppd:.6f}".encode()).hexdigest()[
            :24
        ]
        with self.lock:
            if identity in self.cache:
                self.cache.move_to_end(identity)
                return identity, self.cache[identity], True
            import flip_evaluator

            arrays = []
            for image_id in (first, second):
                with Image.open(self.catalog.assets[image_id]) as image:
                    arrays.append(
                        np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                    )
            if arrays[0].shape != arrays[1].shape:
                raise ValueError(
                    "FLIP needs images with identical dimensions; choose a matching pair"
                )
            heatmap, mean, _ = flip_evaluator.evaluate(
                *arrays, "LDR", inputsRGB=True, applyMagma=True, parameters={"ppd": ppd}
            )
            image = Image.fromarray(
                np.rint(np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
            )
            stream = BytesIO()
            image.save(stream, format="PNG")
            value = {"png": stream.getvalue(), "mean_error": float(mean), "ppd": ppd}
            self.cache[identity] = value
            while len(self.cache) > self.limit:
                self.cache.popitem(last=False)
            return identity, value, False

    def image(self, identity):
        with self.lock:
            return self.cache.get(identity)


def make_server(catalog, port=8765):
    maps = FlipMaps(catalog)

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def json(self, code, value):
            self.send(code, json.dumps(value).encode(), "application/json")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/catalog":
                return self.json(200, {"scenes": catalog.scenes})
            if path.startswith("/asset/"):
                asset = catalog.assets.get(path.removeprefix("/asset/"))
                if asset is None:
                    return self.json(404, {"error": "Image not found"})
                with Image.open(asset) as image:
                    mime = Image.MIME[image.format]
                return self.send(200, asset.read_bytes(), mime)
            if path.startswith("/error/") and path.endswith(".png"):
                value = maps.image(path.removeprefix("/error/").removesuffix(".png"))
                if value is not None:
                    return self.send(200, value["png"], "image/png")
                return self.json(404, {"error": "Recompute this error map"})
            filename = {
                "/": "index.html",
                "/app.js": "app.js",
                "/styles.css": "styles.css",
            }.get(path)
            if filename:
                content_type = {
                    "index.html": "text/html; charset=utf-8",
                    "app.js": "text/javascript; charset=utf-8",
                    "styles.css": "text/css; charset=utf-8",
                }[filename]
                return self.send(200, (UI / filename).read_bytes(), content_type)
            self.json(404, {"error": "Not found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/flip":
                return self.json(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                identity, value, cached = maps.compute(
                    request["a"], request["b"], request["ppd"]
                )
                self.json(
                    200,
                    {
                        "url": f"/error/{identity}.png",
                        "mean_error": value["mean_error"],
                        "ppd": value["ppd"],
                        "cached": cached,
                    },
                )
            except ImportError:
                self.json(
                    503, {"error": "Install .[diagnostics] to enable FLIP error maps"}
                )
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.json(400, {"error": str(error)})

        def log_message(self, fmt, *args):
            if len(args) < 2 or str(args[1]) != "200":
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=Path("rendered/perceptual_experiments/native")
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Add an image, optionally LABEL=PATH",
    )
    parser.add_argument(
        "--ppd", type=float, default=67, help="Viewing density for custom images"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    catalog = RenderCatalog()
    try:
        if not args.image:
            catalog.add_report(args.report)
        else:
            catalog.add_images(args.image, args.ppd)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if not catalog.scenes:
        parser.error("No images found")
    server = make_server(catalog, args.port)
    print(f"Render viewer: http://127.0.0.1:{server.server_port}", flush=True)
    print(
        "Space flips A/B; Enter toggles automatic flipping; F opens fullscreen. Ctrl+C stops the viewer.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
