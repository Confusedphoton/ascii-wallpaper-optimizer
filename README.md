# ASCII Wallpaper Optimizer

This fork of [ASCII Wallpaper](https://gitlab.com/Korvexx/ascii-wallpaper) renders KDE Plasma wallpapers with an offline numerical optimizer. It chooses font glyphs and their foreground colors by scoring the actual rendered pixels, including structure across character cells. Slideshow transitions are sequences of optimized glyph/color grids: one character occupies each cell at every instant, without blending two ASCII images on screen.

The Plasma package only plays prepared assets. Rendering can take time, especially at the 2×4 detail level or for a slideshow. PyTorch can run the solve on a CPU or CUDA GPU.

## Install and prepare a wallpaper

Install the Plasma wallpaper package from this directory:

```sh
python3 scripts/build_plasma_package.py
kpackagetool6 --type Plasma/Wallpaper --install "$PWD/dist/plasma-wallpaper"
```

For subsequent updates, rebuild the package, then use `--upgrade "$PWD/dist/plasma-wallpaper"` instead of `--install`. The staged package contains only Plasma files; the offline renderer stays in the project directory.

Install the Python renderer in a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

If your Python installation does not include `pip`, `uv venv .venv` and `uv pip install --python .venv/bin/python -e .` work as well.

Render at the display's pixel resolution. List slideshow images in their intended order. For one image, pass one path; for a slideshow, pass at least two.

```sh
.venv/bin/ascii-wallpaper-render \
  --size 1920x1080 --detail 1 \
  --font "DejaVu Sans Mono" \
  --frames 16 \
  --output "$HOME/Pictures/ascii-rendered" \
  "$HOME/Pictures/first.jpg" "$HOME/Pictures/second.jpg"
```

In Plasma's wallpaper settings, select **ASCII Wallpaper Optimizer**, select the same source images in the same order, then set **Rendered Assets** to the absolute output directory. Match **Detail Level** and **Transition Frames** to the command. The source image list determines slideshow order; the prepared grids supply the displayed pixels. Re-run the command after changing images, image order, font, characters, detail, grayscale, crop, or resolution. The renderer skips work when its full input signature and outputs match, and reuses unchanged stills and image pairs when possible.

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
