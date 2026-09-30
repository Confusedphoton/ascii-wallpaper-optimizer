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

`compare_glyph_entropy.py` compares an entropy-disabled control with
frame-wide glyph entropy at strengths 0.005, 0.01, 0.02, and 0.06 on all four source
images in `screenshots/`, at cell sizes 12×24 and 8×16. It saves source/before/after
previews, enlarged glyph crops, discrete grids, glyph histograms, color and edge
errors, reproducibility metadata, and a `review.md` with links to each comparison.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_glyph_entropy.py --device cuda
```

Use `--device cpu` without a GPU. `--width`, `--cells`, `--steps`, `--strengths`,
`--dither`, `--chars`, and `--output` override the experiment settings. Every
sweep includes strength zero and holds the glyph bank fixed across its variants.
The current repository bank has M, W, and ´; `--chars` can reproduce a GUI bank
that also includes @. The report tracks @, M, W, ´, &, #, and the literal backtick,
marking absent glyphs as missing rather than treating them as observed zeros.

Headroom rendering defaults to `glyph_entropy_strength=0.01` in the Python
APIs, CLI, and wallpaper GUI. Set `--glyph-entropy-strength 0` to disable it;
the CLI and GUI allow values from 0 to 0.1, and the GUI's Glyph Variety slider
adjusts in steps of 0.001 independently for each profile.
The opt-in term adds `strength × sum(p × ln(p))`, where p is each glyph's share
over the whole frame. During soft optimization, probabilities are aggregated
by actual glyph identity, merging duplicate shortlist/transition candidates.
During discrete refinement, proposals include their exact individual histogram
change; simultaneous proposals are accepted only when the complete regularized
objective improves. Color polishing keeps the chosen histogram fixed. The
existing per-cell entropy penalty still encourages decisive glyph choices.
The new term includes every glyph, including spaces, and targets no particular
character. A higher score for diversity can cost image fidelity, especially in
uniform areas. Compare the actual rendered images before selecting a strength.

Structure-first selection defaults to zero and rejects nonzero strength.
Fixed-glyph transition fitting does not
reselect glyphs or need a histogram objective.

`compare_error_movement.py` evaluates an opt-in local error transport objective
on the blue/yellow tail boundary in `test_image.png` and the blush in
`149642007_p0-无题.jpg`. It compares the unchanged production settings with
different transport strengths at 80 and 120 columns. Source, before, and after
panels include enlarged crops, complete previews, saved grids, and a report.
The additional boundary metric counts green ink in cells containing both blue
and yellow source pixels; its thresholds are a diagnostic heuristic.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_error_movement.py \
  --device cuda --strengths 1 4 12 32 64
```

Use `--device cpu` without a GPU. `--width`, `--cells`, `--steps`, and
`--output` override the comparison settings. The optional Python API argument
`glyph_error_movement_strength` defaults to zero. These experiments now inherit
the production full-bank glyph search. It is experimental and has
no wallpaper setting or production CLI flag. The existing blue-noise dither,
gradient objective, and luminosity bonus remain active in these comparisons.

The objective matches sorted projections of displayed OKLab colors against
the original source in overlapping windows of at most 2×2 glyphs. This is a
sliced squared Wasserstein approximation to local color transport: it allows
color error to change position while preserving the distribution of colors in
each window. Thirteen fixed color directions make the solve deterministic.
The term participates in soft glyph selection, discrete refinement, and final
color fitting. Existing pixel and edge terms still constrain spatial placement.
It adds no scan-order error accumulator. Fixed-glyph transition fitting does
not use this experimental term.

`compare_glyph_search.py` compares six-candidate shortlists with the complete
glyph bank, both without error movement and at strengths 4, 12, and 32. The
Python API option `use_shortlist=False` bypasses ranking entirely: every glyph
gets a logit in every cell and remains available during discrete refinement.
The `candidates` limit and `gradient_shortlist_strength` do not restrict that
mode; the full-frame gradient objective remains active. Endpoint glyphs are
already in the full bank and are not appended again. The production default is
`use_shortlist=False`; the former shortlist is retained only as an explicit
Python API comparison mode. Legacy candidate limits are ignored by production
CLI/service rendering and excluded from cache identities.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_glyph_search.py --device cuda
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_glyph_search.py \
  --device cuda --steps 144 --cells 8x16 --strengths 0 4 12 \
  --output rendered/glyph_search/longer
```

Reports include the original per-cell shortlists, the fraction of final glyphs
outside them, and coverage/histograms in two blush cheek regions. Focus panels
at each strength show source/shortlist/full bank. The `factorial_focus_*.png`
panels compare both search modes with and without error movement. All other
production objectives stay unchanged. More glyphs change softmax normalization
and the assignment of candidate-wise dither fields, even with the same seed
and dither amplitude; this tests the complete change in search behavior.

For a control without logit noise, pass `--dither 0` and a separate `--output`
directory. The runner records the top-two learned logit margins and the fraction
of cell winners changed solely by noise before discrete refinement. Its default
`--dither 0.005` preserves the normal renderer setting. Comparing matched noisy
and zero-dither saved grids also shows whether subsequent refinement retains
those changes.

Each JSON report includes input and font hashes, renderer source hashes, the Git
revision and dirty status, software versions, each solve's effective objectives,
elapsed time, and process memory high-water marks. CUDA solves also record their
peak allocated memory. RSS covers the process lifetime, so use a fresh process
when comparing memory. Record the source and code alongside a report to reproduce
an experiment made with uncommitted changes.

`compare_color_spaces.py` retains its existing filename and explicitly enables
the historical shortlist to compare one and six candidates in OKLab; these
limits do not apply to production rendering. Structure comparisons now call the
production structure renderer; the earlier duplicated linear-RGB prototype and
its obsolete density curve were removed. Palette, dither, brighter-ink, canvas,
and display-transform comparisons remain available. Structure brightness and
direct-display variants are experiments and are labeled in their reports.
