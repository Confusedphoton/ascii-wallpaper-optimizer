# ASCII Wallpaper Optimizer

This fork of [ASCII Wallpaper](https://gitlab.com/Korvexx/ascii-wallpaper) renders KDE Plasma wallpapers with an offline numerical optimizer. It chooses font glyphs and their foreground colors by scoring the actual rendered pixels, including structure across character cells. Slideshow transitions are sequences of optimized glyph/color grids: one character occupies each cell at every instant, without blending two ASCII images on screen.

The Plasma package displays prepared assets. A small local service starts rendering when images or appearance settings change. It runs PyTorch in a separate process, so opening wallpaper settings does not load the numerical runtime into Plasma. Rendering can take time, especially at the 2×4 detail level or for a slideshow. PyTorch can run the solve on a CPU or CUDA GPU.

## Install and prepare a wallpaper

Install the Python renderer in a virtual environment, then install the Plasma package and user render service:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/install.py
```

If your Python installation does not include `pip`, `uv venv .venv` and `uv pip install --python .venv/bin/python -e .` work as well. Run the installer again after updating the project. It upgrades the Plasma package and restarts the local user service. Reopen wallpaper settings to load the updated QML.

In Plasma's wallpaper settings, select **ASCII Wallpaper Optimizer** and choose an image or slideshow. Rendering starts automatically. A progress bar and status message appear in the settings and on the desktop while the source image is shown. The generated asset directory is recorded in **Render Cache**. **Render now** retries a failed render or rebuilds completed assets. Adjust font, characters, detail, grayscale, transition frames, and dithering strength in **Appearance**; each change schedules another render.

You can also run the renderer from a terminal for export or batch work:

```sh
.venv/bin/ascii-wallpaper-render \
  --size 1920x1080 --detail 1 \
  --font "DejaVu Sans Mono" \
  --frames 16 \
  --output "$HOME/Pictures/ascii-rendered" \
  "$HOME/Pictures/first.jpg" "$HOME/Pictures/second.jpg"
```

List slideshow images in order. The CLI skips work when its full input signature and outputs match, and reuses unchanged stills and image pairs when possible.

The rendered directory contains `atlas.png`, `metadata.png`, still grids, transition frame grids, still PNG previews, and `manifest.json`. It can be moved as a unit. The preview PNGs are exports; Plasma displays the compact glyph/color maps through a small shader. Each transition includes both exact still endpoints and optimized intermediate frames. The last slideshow image also transitions to the first.

## Quality controls

- `--detail 0`, `1`, `2` uses 2×4, 4×8, or 8×16 pixel cells. `--cell WIDTHxHEIGHT` overrides this; the prepared asset's metadata controls playback cell size.
- `--chars ' ...'` sets the exact glyph set. Include a literal leading space if desired. Characters must be unique, and the Plasma character list must match the rendered assets.
- `--crop fill` fills the display by cropping; `--crop fit` letterboxes onto black. `--grayscale` converts the source before optimization.
- `--dither` controls deterministic, spectrally filtered blue noise in glyph selection; `0` disables it. `--seed` makes the result reproducible.
- `--steps`, `--transition-steps`, and `--candidates` trade rendering time and memory for search quality. `--device cpu` works without a GPU; `--device cuda` uses a supported PyTorch CUDA installation.
- `--force` rebuilds even if the full input signature matches.

## How the optimizer works

The renderer rasterizes each glyph from the selected font at final cell size. It fits a foreground color in linear light for each candidate, then optimizes candidate weights and colors against a full-frame objective with pixel, multiscale color, and edge terms. This full-frame step includes neighboring cells, so tiles are never independently finalized. A final color pass runs with discrete glyphs. The character mask is applied once, avoiding the old shader's double reduction of brightness.

For each slideshow pair, the offline solver starts and ends at the exact optimized stills. Intermediate grids are optimized with temporal shape and color penalties plus a blue noise change schedule. Plasma chooses the prepared frame and draws its glyphs; it performs no pixel crossfade or optimization at display time.

The project remains licensed under [AGPL-3.0-or-later](LICENSE.md).
