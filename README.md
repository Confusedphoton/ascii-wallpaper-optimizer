# ASCII Wallpaper Optimizer

This fork of [ASCII Wallpaper](https://gitlab.com/Korvexx/ascii-wallpaper) renders KDE Plasma wallpapers with an offline numerical optimizer. It chooses font glyphs and their foreground colors by scoring the actual rendered pixels, including structure across character cells. Slideshow transitions are sequences of optimized glyph/color grids: one character occupies each cell at every instant, without blending two ASCII images on screen.

The Plasma package displays prepared assets. A small local service starts rendering when images or appearance settings change. It runs PyTorch in a separate process, so opening wallpaper settings does not load the numerical runtime into Plasma. Rendering can take time, especially with a dense grid or a slideshow. PyTorch can run the solve on a CPU, an NVIDIA CUDA GPU, or an AMD ROCm GPU.

## Install and prepare a wallpaper

Install the Python renderer in a virtual environment, then install the Plasma package and user render service. The setup command checks Linux PCI display devices and asks `uv` to select a compatible CUDA or ROCm PyTorch build when it finds an NVIDIA or AMD GPU. It selects CPU PyTorch if it finds neither:

```sh
python3 scripts/setup_environment.py
.venv/bin/python scripts/install.py
```

The helper uses `uv pip --torch-backend` for hardware-aware selection; plain `uv sync` does not accept that selector. Run the setup command again after updating the project to reinstall the selected PyTorch build. Pass `--backend rocm` to force the ROCm 7.2 wheels on Linux, or use `--backend cpu` or `--backend cuda` to override automatic selection. The installer upgrades the Plasma package and restarts the local user service. On an upgrade it also restarts the running Plasma shell so wallpapers load the new QML and shader; the desktop and panels briefly reload. Reopen wallpaper settings afterward.

If you need to use `pip`, first install the PyTorch wheel for your GPU, then install the project:

```sh
python3 -m venv .venv
```

Choose and run the PyTorch command for your hardware:

| Hardware | PyTorch install command |
| --- | --- |
| AMD GPU on Linux | `.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/rocm7.2` |
| NVIDIA GPU | `.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu132` |
| CPU only | `.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` |

Then install the project and Plasma package:

```sh
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/install.py
```

Installing PyTorch first prevents pip from choosing its default CUDA build on an AMD system.

In Plasma's wallpaper settings, select **ASCII Wallpaper Optimizer** and choose an image or slideshow. Rendering starts automatically. A progress bar and status message appear in the settings and on the desktop while the last saved ASCII render remains visible. The generated asset directory is recorded in **Render Cache** and shown on startup while the renderer checks or rebuilds the current render. **Render now** retries a failed render or rebuilds completed assets. Adjust render style, ink polarity, and grayscale under **Style**; choose a font, characters, detail, dithering, and background under **Appearance**. Monospaced fonts installed on the system are discovered for the font picker. Slideshow profiles also show a **Transition** section for slide interval and transition settings. Each render setting schedules another render. A nonzero glyph luminosity bonus automatically enables chrominance correction.

Use **Editing Profile** to set an independent image or slideshow and appearance settings for profiles A and B. The **Profile Mode** section controls automatic time-of-day switching or a manual choice. Turn on **Switch profiles automatically** and set the two `HH:MM` start times to follow local time of day. When it is off, choose **Manual Profile** A or B. Add **ASCII Wallpaper Profile** to a Plasma panel or dock after installation: click its compact button to pin A or B, then hold it to open the choices. In the menu, click A or B to force that profile; click the selected profile again, or select **Follow schedule**, to clear the override. The override is shared by wallpapers using this plugin and persists across renderer restarts. A newly selected profile may need time to render its assets.

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

- `--detail 0` through `4` map to about 180, 120, 80, 240, and 320 columns on a 2560-pixel-wide display, respectively. The settings menu presents them from ultra high down to low detail. Cell dimensions scale with the monitor's physical resolution, keeping character density similar across displays. The two denser presets increase render time and memory use. `--cell WIDTHxHEIGHT` overrides this; the prepared asset's metadata controls playback cell size.
- `--chars ' ...'` sets the exact glyph set. Include a literal leading space if desired. The shared default includes `\=^`; structure style adds `-_/|+[]{}` automatically when the default set is selected. Characters must be unique, and the Plasma character list must match the rendered assets.
- `--style headroom` (default) uses the full-frame glyph and color solver with a local, chroma-preserving luminance boost. `--style structure` selects glyphs from coverage and within-cell texture with a stronger tone curve and blue-noise dither. Its default characters add `-_/|+[]{}`. The settings switch to this set automatically when the current list is still the standard default; custom lists stay intact.
- `--invert-ink` selects glyphs for dark ink on a light canvas. Inversion participates in solving and is recorded in the rendered assets, so exported previews and Plasma playback use the same polarity. It can be combined with either style.
- `--crop fill` fills the display by cropping; `--crop fit` letterboxes onto black. `--grayscale` converts the source before optimization.
- `--dither` controls deterministic blue noise for the headroom solver; `--structure-dither` controls coverage dithering for the structure solver (default `0.06`). `--gradient-loss-strength` controls Sobel edge matching in displayed OKLab (default `1.5` for headroom, `0` for structure). `--glyph-luminance-bonus` favors lighter ink (default `0.12` for headroom); its chrominance correction is automatically enabled whenever the bonus is nonzero. These strengths are unavailable for structure style. `0` disables a loss. `--seed` makes the result reproducible.
- `--background auto` uses one dark color derived from the image; `--background '#202028'` fixes the color. Plasma offers the same modes and a color picker. Every still and transition frame has one uniform background color.
- `--steps`, `--transition-steps`, and `--candidates` trade rendering time and memory for search quality. `--device cpu` works without a GPU; `--device cuda` uses a supported NVIDIA CUDA or AMD ROCm PyTorch installation.
- `--transition-pace` chooses linear, ease-in, ease-out, or ease-in-out progress between stills. `--transition-field` chooses blue-noise, ripple, spiral, directional, edge, or salience timing across cells; `--transition-spread` controls how far apart cells begin changing. Wallpaper settings also control the playback duration in milliseconds. Increase transition frames if the selected endpoints need more steps to traverse their glyph paths.
- `--force` rebuilds even if the full input signature matches.

## How the optimizer works

The renderer rasterizes each glyph from the selected font at final cell size. The headroom style fits a foreground color in linear light for each candidate, then optimizes candidate weights and colors against a full-frame OKLab objective with pixel, multiscale color, and lightness-edge terms. This perceptual objective is the default; linear RGB remains available to the solver as an explicit option. The full-frame step includes neighboring cells, so tiles are never independently finalized. A final color pass runs with discrete glyphs. The structure style scores glyph coverage and within-cell OKLab lightness texture, then uses one bright ink color per cell. Both styles use one uniform background, chosen from the image's darkest areas or set explicitly, and apply the same headroom-aware output boost once. The character mask is applied once, avoiding the old shader's double reduction of brightness.

For each slideshow pair, the offline solver starts and ends at the exact optimized stills. It builds a font-specific distance graph from rasterized glyph masks and follows shortest paths between each cell's endpoint glyphs. Each cell receives a phase offset from the selected spatial field, including a small blue-noise component for structured fields. Intermediate colors are fitted to the resulting glyphs. Plasma chooses the prepared frame and draws its glyphs; it performs no pixel crossfade or optimization at display time.

The project remains licensed under [AGPL-3.0-or-later](LICENSE.md).
