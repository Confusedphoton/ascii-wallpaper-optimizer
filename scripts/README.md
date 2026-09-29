The `compare_*.py` scripts create visual comparisons and JSON metrics using the
current renderer. Run them from the repository root with the project's virtual
environment. Most use the screenshot fixtures and constants near the top of the
file; `compare_color_strategies.py` also accepts a source image, output directory,
size, cell size, and optimization steps as command-line options.

```sh
.venv/bin/python scripts/compare_color_strategies.py example.png \
  --output rendered/comparison --size 960x540 --cell 8x16 --steps 36
```

Shared helpers in `experiment_utils.py` keep grid colors in OKLab, round-trip ink,
background, and masks through playback's RGB8 assets, then apply the display
transform. Metrics measure the saved RGB8 preview. Dense-glyph fractions average
mask pixels within each cell. Comparisons inherit production objective defaults
unless the experiment explicitly overrides them.

Each JSON report includes input and font hashes, renderer source hashes, the Git
revision and dirty status, software versions, each solve's effective objectives,
elapsed time, and process memory high-water marks. CUDA solves also record their
peak allocated memory. RSS covers the process lifetime, so use a fresh process
when comparing memory. Record the source and code alongside a report to reproduce
an experiment made with uncommitted changes.

`compare_color_spaces.py` retains its existing filename but compares candidate
counts in the current OKLab objective. Structure comparisons now call the
production structure renderer; the earlier duplicated linear-RGB prototype and
its obsolete density curve were removed. Palette, dither, brighter-ink, canvas,
and display-transform comparisons remain available. Structure brightness and
direct-display variants are experiments and are labeled in their reports.
