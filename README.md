# ASCII Wallpaper Optimizer

This fork of [ASCII Wallpaper](https://gitlab.com/Korvexx/ascii-wallpaper) renders KDE Plasma wallpapers with an offline numerical optimizer. It chooses font glyphs and their foreground colors by scoring the actual rendered pixels, including structure across character cells. Slideshow transitions are sequences of optimized glyph/color grids: one character occupies each cell at every instant, without blending two ASCII images on screen.

The Plasma package displays prepared assets. A small local service starts rendering when images or appearance settings change. It runs PyTorch in a separate process, so opening wallpaper settings does not load the numerical runtime into Plasma. Rendering can take time, especially with a dense grid or a slideshow. PyTorch can run the solve on a CPU, an NVIDIA CUDA GPU, or an AMD ROCm GPU.

## Install and prepare a wallpaper

Install the Python renderer in a virtual environment, then install the Plasma package and user render service. The setup command checks Linux PCI display devices, selects ROCm PyTorch for AMD GPUs, and asks `uv` to select a compatible CUDA build for NVIDIA GPUs. It selects CPU PyTorch if it finds neither. After installation it verifies a computation and backpropagation on the selected device:

Packaging requires the Qt 6 Shader Tools `qsb` compiler. The builder finds it through `PATH`, Qt's installation directory, or common Linux Qt locations; set `QSB=/path/to/qsb` to override discovery. Every build compiles the current shader sources before replacing the staged package. Compiled shaders are generated in `dist/`, rather than stored alongside the sources.

On Linux systems with `/dev/accel` NPU devices, install `bubblewrap` as well. The numerical launcher hides those devices from each optimizer process while preserving GPU access through `/dev/dri` and `/dev/kfd`. This avoids ROCm initialization hanging in the unrelated AMD NPU driver. The service, setup verification, and `ascii-wallpaper-render` command use the launcher automatically; for direct module calls, use `.venv/bin/python -m ascii_optimizer.launcher -m ascii_optimizer.cli ...`.

```sh
python3 scripts/setup_environment.py
.venv/bin/python scripts/install.py
```

The helper uses `uv pip --torch-backend` for hardware-aware selection; plain `uv sync` does not accept that selector and can replace your hardware-specific wheel. Run the setup command again after updating the project to reinstall the selected PyTorch build. It preserves an existing virtual environment. Adding a ROCm library path cannot enable a CPU-only PyTorch build. Pass `--backend rocm` to force the ROCm 7.2 wheels on Linux, or use `--backend cpu` or `--backend cuda` to override automatic selection. The installer upgrades the Plasma package and restarts the local user service. On an upgrade it also restarts the running Plasma shell so wallpapers load the new QML and shader; the desktop and panels briefly reload. Reopen wallpaper settings afterward.

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

In Plasma's wallpaper settings, select **ASCII Wallpaper Optimizer** and choose an image or slideshow. Rendering starts automatically. A progress bar and status message appear in the settings and on the desktop while the last saved ASCII render remains visible. The generated asset directory is recorded in **Render Cache**. Each profile also saves the completed generation’s manifest and playback settings, so startup restores its frames immediately without waiting for the render service or loading PyTorch. The service checks the current settings in the background; the cached wallpaper remains visible while it reconnects or renders a replacement. **Render now** retries a failed render or rebuilds completed assets. Clients retry automatically when the service is temporarily unavailable and check completed jobs for new generations, so a forced render also refreshes an already running wallpaper. Adjust render style, ink polarity, and grayscale under **Style**; choose a font, characters, detail, dithering, and background under **Appearance**. Monospaced fonts installed on the system are discovered for the font picker. Slideshow profiles also show a **Transition** section for slide interval and transition settings. Each render setting schedules another render. A nonzero glyph luminosity bonus automatically enables chrominance correction.

Use **Editing Profile** to set an independent image or slideshow and appearance settings for profiles A and B. The editor remembers the last profile you opened, even when you close without applying; **Source** appears directly below the profile controls. Apply or OK saves each profile’s source list and render settings. Edit the complete character set in one text field, including any literal space, or choose **Restore defaults**. The **Profile Mode** section controls automatic time-of-day switching or a manual choice. Turn on **Switch profiles automatically** and set the two `HH:MM` start times to follow local time of day. When it is off, choose **Manual Profile** A or B. Add **ASCII Wallpaper Profile** to a Plasma panel or dock after installation: click its compact button to pin A or B, then hold it to open the choices. In the menu, click A or B to force that profile; click the selected profile again, or select **Follow schedule**, to clear the override. The override is shared by wallpapers using this plugin and persists across renderer restarts. A newly selected profile may need time to render its assets.

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

The rendered directory contains `atlas.png`, `metadata.png`, still grids, transition frame grids, still PNG previews, and `manifest.json`. It can be moved as a unit. The preview PNGs are exports; Plasma displays the compact glyph/color maps through a small shader. The settings thumbnail composes at full display resolution before downsampling, preserving glyph coverage and apparent colors. Embedded image color profiles are converted to sRGB before solving. Each transition includes both exact still endpoints and optimized intermediate frames. The last slideshow image also transitions to the first. Playback loads a complete replacement frame into a second buffer and keeps the current frame visible until its replacement is ready; transition timing pauses if a needed frame is still loading.

Playback advances only at prepared-frame boundaries, including when a frame is held for several seconds. Each complete frame is composed once into a GPU texture, which stays frozen between frame changes and can be reused when other desktop items redraw. Display-size changes and graphics-context recovery refresh that texture. The RGBA output texture uses four bytes per physical display pixel (about 28 MiB at 3414×2134), plus Qt's render-target buffers and the compact input maps; it does not load the PNG previews for playback.

The service publishes complete renders in immutable generation directories and maintains a separate content-based still cache. Changing transition settings reuses matching stills without overwriting another variant. Cache signatures include the renderer source revision, so local color or solver fixes invalidate old variants and stills even between releases. `--still-cache PATH` enables the same reuse for CLI exports. A forced render publishes a new generation and refreshes the still cache atomically. Requests superseded by the same client are removed from the queue or stopped when no other client needs them. Source-image memory stays bounded to a single still or two images per active transition worker.

## Quality controls

- `--detail 0` through `4` map to about 180, 120, 80, 240, and 320 columns on a 2560-pixel-wide display, respectively. The settings menu presents them from ultra high down to low detail. Cell dimensions scale with the monitor's physical resolution, keeping character density similar across displays. The two denser presets increase render time and memory use. `--cell WIDTHxHEIGHT` overrides this; the prepared asset's metadata controls playback cell size.
- `--chars ' ...'` sets the exact glyph set. Include a literal leading space if desired. The shared default includes `#\/=^|[]{}`; structure style adds `-_+` automatically when the default set is selected. Characters must be unique, and the Plasma character list must match the rendered assets.
- `--style headroom` (default) uses the full-frame glyph and color solver with OKLab color fitting and a local, chroma-preserving luminance boost. `--style structure` selects glyphs from coverage and within-cell OKLab texture with a stronger tone curve and blue-noise dither. Its default characters add `-_+`. The settings switch to this set automatically when the current list is still the standard default; custom lists stay intact.
- `--invert-ink` selects glyphs for dark ink on a light canvas. Inversion participates in solving and is recorded in the rendered assets, so exported previews and Plasma playback use the same polarity. It can be combined with either style.
- `--crop fill` fills the display by cropping; `--crop fit` letterboxes onto black. `--grayscale` converts the source before optimization.
- `--dither` controls post-optimization, candidate-wise logit noise for the headroom solver (default `0.005`; the default blue-noise uniform variant has logit standard deviation about `0.144`); `--structure-dither` controls coverage dithering for the structure solver (default `0.06`). `--gradient-loss-strength` controls Sobel edge matching in displayed OKLab (default `1.5` for headroom, `0` for structure). `--glyph-luminance-bonus` favors lighter ink with a reward that tapers near white (default `0.12`, adjustable through `0.5` for headroom); its chrominance correction is automatically enabled whenever the bonus is nonzero. `--glyph-entropy-strength` rewards frame-wide glyph variety (default `0.01`, adjustable from `0` to `0.1`); the GUI exposes it as a Glyph Variety slider for each profile. These strengths are unavailable for structure style. Setting a strength to `0` disables that feature. `--seed` makes the result reproducible.
- `--background auto` uses one dark color derived from the image; `--background '#202028'` fixes the displayed color, precompensated for the display transform and ink polarity. Plasma offers the same modes and a color picker. Every still and transition frame has one uniform background color. Ink and backgrounds are constrained to the saved RGB8 palette; previews compose those exact saved colors.
- `--steps` and `--transition-steps` trade rendering time for search quality. The headroom optimizer considers every configured glyph in every cell; the legacy `--candidates` argument is accepted but ignored. A larger character set increases solver memory use. `--device auto` (default) selects an available GPU; `--device cpu` works without a GPU; `--device cuda` uses a supported NVIDIA CUDA or AMD ROCm PyTorch installation. The CLI and service journal report the selected device and PyTorch build.
- `--transition-workers` runs independent slideshow pairs concurrently (default `2`). GPU workers use separate CUDA/ROCm streams. Frames within each pair retain their ordered glyph paths and previous-frame color regularization. Use `1` to reduce memory use, or increase it when memory allows; changing worker count does not invalidate cached assets. Set `ASCII_WALLPAPER_TRANSITION_WORKERS` in the service's environment to override its default.
- `--transition-pace` chooses linear, ease-in, ease-out, or ease-in-out progress between stills. `--transition-field` chooses blue-noise, ripple, spiral, directional, edge, or salience timing across cells; `--transition-spread` controls how far apart cells begin changing. Wallpaper settings also control the playback duration in milliseconds. Increase transition frames if the selected endpoints need more steps to traverse their glyph paths.
- `--force` rebuilds even if the full input signature matches.

## How the optimizer works

The renderer rasterizes each glyph from the selected font at final cell size. The headroom style optimizes candidate glyph weights and scores colors in OKLab, using pixel, multiscale color, lightness-edge, and displayed-gradient terms. Ink parameters use bounded sRGB channels with RGB8 rounding during scoring, so fitting follows the actual displayable gamut without repeatedly stripping chroma from OKLab updates. The automatic chrominance correction measures saturation along the source hue and penalizes chroma in other directions. When displayed lightness increases, the required chroma increases with it to preserve the source chroma-to-lightness ratio; darker pixels retain the source chroma floor so stronger ink can compensate for glyph gaps. The displayed-color gradient accounts for gamut compression, so the solver can see saturation lost at a color’s brightness limit. The lighter-ink reward has a marginal gain of `(1 - L)²`, retaining shadow lift while easing off near white; empty glyphs receive no reward. Every configured glyph remains available in every cell throughout optimization and discrete refinement. Initial fitted ink and logits use OKLab pixel error, chrominance correction, and the optional light-ink bonus; displayed OKLab Sobel gradients participate in the full-frame objective. The full-frame step includes neighboring cells, so tiles are never independently finalized. Glyph identity is represented by one learnable logit per configured glyph; a temperature-scaled softmax blends their masks during optimization, then argmax and discrete full-frame refinement choose the final character. Post-optimization noise perturbs the learned logits before argmax. A final color pass runs with those discrete glyphs and retains its best saved-color fit. The structure style scores glyph coverage and within-cell OKLab lightness texture, then uses one bright ink color per cell. Both styles use one uniform OKLab background, chosen from the image's darkest areas or set explicitly, and apply the same headroom-aware output boost once. The character mask is composed once in OKLab.
Headroom noise is applied only after optimization, directly to the learned candidate logits before choosing discrete glyphs. The default uses bounded uniform values from deterministic blue-noise rank fields, one field per candidate. The Python API also supports IID Gumbel noise via `logit_noise_distribution="gumbel"`; it draws independent PyTorch uniforms and transforms them with `-log(-log(u))`. At unit scale, argmax is the exact Gumbel-Max sample from the learned softmax distribution; other scales temper that distribution. The two variants have different variances. The subsequent full-frame discrete refinement is deterministic and can revise the selected glyphs. Run `.venv/bin/python scripts/compare_logit_noise.py` to regenerate the four-image comparison in `rendered/logit_noise/`.

An experimental Python API option, `glyph_error_movement_strength=0.0`, adds local color transport during glyph and color optimization. It matches color distributions in overlapping 2×2 glyph neighborhoods, allowing nearby glyphs to compensate for color error without requiring a mixed hue in every cell. It is disabled by default and awaits visual review. Run `.venv/bin/python -m ascii_optimizer.launcher scripts/compare_error_movement.py --device cuda --strengths 1 4 12 32 64` for tail-boundary and blush comparisons; use `--device cpu` without a GPU. See [the experiment instructions](scripts/README.md) for the method and controls.

Glyph entropy regularization defaults to `glyph_entropy_strength=0.01` for headroom rendering. It rewards variety in the frame-wide glyph histogram while retaining decisive per-cell choices, and participates in soft selection and discrete refinement without singling out particular glyphs. Adjust it with the GUI’s Glyph Variety slider or `--glyph-entropy-strength`; zero disables it. Structure rendering keeps it disabled. Run `.venv/bin/python -m ascii_optimizer.launcher scripts/compare_glyph_entropy.py --device cuda` to generate comparisons on all four source images, enlarged crops, histograms, and a review report; use `--device cpu` without a GPU. See [the experiment instructions](scripts/README.md) for controls and the fidelity tradeoff.

The headroom solver defaults to `use_shortlist=False`: all glyphs participate in both the soft solve and discrete refinement. Shortlisting remains available only through an explicit Python API comparison mode, `use_shortlist=True`. Run `.venv/bin/python -m ascii_optimizer.launcher scripts/compare_glyph_search.py --device cuda` to compare the former six-glyph shortlist with the full bank, with and without experimental error movement. Cache identities record the full-bank search and renderer source revision, so existing shortlisted assets rebuild automatically.

For each slideshow pair, the offline solver starts and ends at the exact optimized stills. It builds a font-specific distance graph from rasterized glyph masks and follows shortest paths between each cell's endpoint glyphs. Each cell receives a phase offset from the selected spatial field, including a small blue-noise component for structured fields. Salience fields aggregate detail over each cell. The renderer selects path glyphs first, then fits their colors with glyph identities fixed; `--transition-steps` controls this retained color optimization for headroom style. Structure style assigns its bright per-cell inks directly. Constant target gradients, color pyramids, and chroma weights are prepared once per solve, and candidate batches have bounded pixel workloads. Plasma chooses the prepared frame and draws its glyphs; it performs no pixel crossfade or optimization at display time. Manifest uniforms provide display dimensions and each frame's background without decoding constant metadata for every output pixel.

## Development checks

Install the development dependency group into the existing hardware-configured environment, then run the tests and build the packages:

```sh
uv pip install --python .venv/bin/python --group dev
.venv/bin/python -m ascii_optimizer.launcher -m pytest -q
.venv/bin/python scripts/build_plasma_package.py
```

The suite covers saved-color parity, transition paths and color fitting, immutable cache generations, service requests and recovery, and packaging. Qt Quick lifecycle tests run when `qmltestrunner` and the Qt Quick Test module are available. The comparison scripts share playback preview and metric helpers; their reports record input/font hashes, renderer revision, effective options, and measured solve costs.

The GPU output-cache checks use Qt's offscreen backend. Additional 2× display-scale cases require an exposed X11 window and can be enabled with `ASCII_WALLPAPER_TEST_NATIVE_QML=1`; leave this unset for headless runs.

The project remains licensed under [AGPL-3.0-or-later](LICENSE.md).
