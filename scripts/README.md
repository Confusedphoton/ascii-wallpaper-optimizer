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
over the whole frame. During glyph optimization, hard forward probabilities are aggregated
by actual glyph identity, merging duplicate shortlist/transition candidates.
During discrete refinement, proposals include their exact individual histogram
change; simultaneous proposals are accepted only when the complete regularized
objective improves. Color polishing keeps the chosen histogram fixed. Forward selection is already discrete, so there is no per-cell mixture entropy penalty.
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
The term participates in hard forward glyph selection, discrete refinement, and final
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

`compare_discrete_layout.py` compares the previous renderer with exact contextual
refinement using soft forward selection, hard forward selection, and hard
selection with the experimental cell-aligned patch loss. It uses the existing
tail-boundary and blush fixtures, at 12×24 and 8×16 cells by default. Every
replacement includes neighboring pixels and affected globally aligned pooling
blocks; accepted batches must improve the complete discrete objective. Ink is
refitted during repeated refinement and after glyph selection.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_discrete_layout.py \
  --device cuda --baseline-ref <commit-before-the-fix>
```

Before committing the fix, `--baseline-ref HEAD` selects the previous optimizer.
Alternatively, pass `--baseline-file /path/to/optimizer.py`. The script preserves
that source as `optimizer_before.py` and the current renderer as
`optimizer_after.py`, with their hashes in the report. Use `--width`,
`--cells`, `--steps`, `--layout-strength`, and `--output` to control the experiment;
use `--device cpu` without a GPU. The patch term is available only through the
Python `solve` API as `layout_loss_strength`, with production default zero.
The report compares every variant under the same non-layout discrete objective,
and records RGB8 playback color/edge errors, focus crops, runtime, and per-solve
GPU peak allocation. Pass `--variants before hard_exact` for a direct
production before/after comparison. Hard forward selection and exact refinement increase the
search work; runtime is part of the comparison, rather than an assumed constant.

For performance comparisons, supply a second implementation with
`--reference-file /path/to/optimizer_previous.py --variants before reference hard_exact`.
The baseline and reference are preserved separately as `optimizer_before.py`
and `optimizer_reference.py`; `optimizer_after.py` records the current solver.
All three use the same production settings and serialized playback metrics.

The current discrete scorer prepares residual/adjoint convolution fields once
per cell group and computes exact quadratic loss deltas from changed pixels.
Candidate edge kernels process small delta halos; pooling uses the global block
alignment. Display mapping uses an exact lookup over actual atlas coverage
values for repeated current-ink colors. Unchanged analytic display colors may
be cached within a 64 MiB budget. Proposed inks are fitted in larger bounded
batches. These changes preserve the scoring objective and the full glyph bank;
small floating-point differences can change tied or nearly tied choices.

`trace_solver.py` records the full optimization trajectory and renders stage
comparisons for the tail and blush fixtures. Install optional plotting support
with `uv pip install --python .venv/bin/python '.[diagnostics]'`, then run:

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/trace_solver.py \
  --device cuda --steps 36 --step-budgets 0 36 72 \
  --refinement-sweeps 2 --refinement-budgets 0 1 2 4 --polish-steps 9 \
  --timing-repeats 3 --output rendered/solver_diagnostics/saturation_fixed
```

The Python API accepts an optional `SolverTrace` through `solve(...,
diagnostics=trace)` or headroom `solve_style(..., diagnostics=trace)`. Call
`trace.save(directory)` after solving. Ordinary rendering records nothing.
Diagnostics keep detached snapshots on their existing device during solving
and transfer them afterward; GPU event timing avoids per-record synchronization.
The trace adds memory and evaluation overhead, so the script separately times
ordinary solves and performs optional profiler runs over sampled Adam and
refinement intervals. Keep other GPU jobs idle for timing comparisons.

Each fixture writes raw errors, coefficients, weighted contributions, sampled
weighted gradient RMS in the actual Adam parameters, and phase intervals in
`trace.json`, full per-cell/per-candidate logits with identities, temperatures,
and actual selection noise in `logits.npz`, and every evaluated glyph/color grid
in `grids.npz`. Logits are saved before each Adam update and once after the last;
they stop changing before discrete refinement. Fixed-temperature entropy curves
distinguish learned concentration from temperature annealing. Intermediate
previews separate the final Adam iterate, post-dither
selection, pre-refinement state, every sweep, and final color polish. Histograms,
coverage, and ink lightness also cover the lowest-variance quarter of source
cells, using the same region mask at every stage. Acceptance counters expose
batch subdivision costs. Each step budget has its own temperature schedule,
with refinement and final color polish held fixed independently. Normal rendering
disables refinement (`refinement_sweeps=0`) and keeps nine color-polish updates;
the trace script uses two sweeps by default to expose intermediate refinement
states for comparison. The default
ablation also compares zero, one, two, and four refinement sweeps at the same
Adam budget. Ordinary solves repeat three times; tables show medians and plots
show the observed timing ranges. Adam-only runs are calibrated against each
nonzero refinement budget; the report explicitly flags matches outside 10%.
Use `--no-matched-time` to skip calibration or `--timing-repeats 1` for a quick
inspection. Instrumented trace times include the extra diagnostic work.

`adam_comparison.png` and `adam_focus.png` include the source, actual
initialization, raw snapshots every six updates, the raw terminal iterate, and
the returned frame after dither and polish. Their metrics are recorded
separately in `analysis.adam_stages`; a flat returned-frame budget curve does
not establish that raw Adam iterates stopped improving visually. The weighted
objective and displayed pixel/Sobel errors are optimization proxies, rather
than validated perceptual quality measures.

Use `--images viviana --cell 8x16 --refinement-sweeps 0
--refinement-budgets 0 --no-matched-time` to inspect Vivian; this additional
fixture is selected explicitly. `--glyph-entropy-strength 0` provides an
entropy-disabled control with otherwise identical settings. Its objective
excludes the entropy reward, so total scores across the two settings are not
directly comparable.

`visual.npz` stores per-cell probability-weighted pixel and oriented-edge
variance in composed encoded OKLab, at both scheduled and fixed temperature,
and the actual raster/edge distance between the two leading glyphs. These
geometry metrics hold ink fixed and omit display tone/gamut mapping. They do
not treat glyph IDs as ordered. Raw logit margins remain available as selection
sensitivity diagnostics; they are not calibrated confidence or visual distance.
`replacements_pre_refinement.npz` and `replacements_post_refinement.npz` store
exact single-cell whole-objective deltas for every available glyph with current
RGB8 ink held fixed. These include neighbors and affected multiscale regions;
individual deltas cannot be added to predict interacting simultaneous moves.

Saturation now uses a symmetric, source-referenced `C/L` hue error, without a
one-sided deficit or darker-pixel chroma floor. Black source chroma is exactly
zero. The raw error includes the documented bounded source importance weights
(1–5); the separately logged coefficient defaults to 20 with the light-ink
bonus. Other terms have their own reductions: most are squared OKLab errors,
glyph entropy is in nats, and rewards are negative. Raw numeric values alone
are not interchangeable measures of optimization influence, which is why the
trace also samples each weighted term's parameter derivatives. Those samples
include the scheduled temperature and straight-through derivatives and add
cost only to explicitly instrumented runs.

Batch acceptance now caches both unchanged and proposed display pixels, since
tone/gamut mapping is pixel-local. Each trial still evaluates the full image
objective and interactions, while avoiding repeated gamut mapping of the same
frame. The Adam loop evaluates actual hard frames and contracts glyph weights with
the shared bank. Terminal layouts are never rolled back to initialization or
an earlier iterate. Fixed-size histogram accumulation and precomputed parity groups
avoid output-shape synchronization.

`compare_perceptual.py` compares spatial-lightness and contour
variants, with independent official FLIP evaluation. Its `combined_add` variant
is the historical perceptual baseline; production now also calibrates the
atlas display range and scores chromatic raster stroke alignment. The script explicitly disables the
production perceptual terms before adding each variant, so its legacy baseline
stays comparable and losses are not counted twice. It reads the installed
profiles and copies their completed previews as references, without changing
settings or wallpaper caches. Current-code baselines are separate because a
cached preview can come from an earlier renderer revision.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_perceptual.py \
  --device cuda --width 3414 \
  --variants baseline brightness_2 tensor combined_add combined_tensor_add combined_strokes \
  --output rendered/perceptual_experiments/native
.venv/bin/python scripts/plot_perceptual.py rendered/perceptual_experiments/native
```

The viewing hypothesis is a 13-inch 16:10 display at two feet. Image samples
per visual degree account for the image's displayed width, including renders
resampled to physical screen pixels. Nominal and ±20% viewing scales are
reported. The 500-nit peak capability does not specify actual SDR white,
display black, or ambient adaptation; no absolute brightness calibration is
inferred. Optional `.[diagnostics]` dependencies include the FLIP evaluator.

Spatial lightness integrates displayed linear-RGB luminance before CIE L*,
at Gaussian standard deviations of 2.5/5/10 arcminutes. It measures source
preservation with symmetric error and fixed shadow importance. Source-defined
lightness bins expose shadow lift alongside darker midtones. This is an
S-CIELAB-inspired luminance proxy, rather than the complete calibrated model.

Contour metrics prefilter luminance and compare unsigned orientation tensors
or 12-bin local direction histograms over overlapping windows two/four glyph
heights wide. Circular Wasserstein distance uses cumulative histogram
differences minus their median, so wraparound and neighboring orientations
have proper angular costs. It is exact for the discretized histograms, with
soft angular bin assignment. `combined_strokes` compares finer rendered
strokes against coarser source direction fields. Source-edge strength matching
penalizes disappearing contours. These metrics encourage local directional
agreement; they do not guarantee connected outlines or preserve curve topology.

All perceptual experiments disable refinement: its quadratic replacement cache
does not implement these nonlinear losses. Production diagnostics automatically
skip unsupported replacement sweeps. `SolverTrace(include_replacements=False)`
skips that diagnostic sweep while retaining losses, logits, and state grids.
Production uses the approved terms directly from `ascii_optimizer.perceptual`.
Script-local objective wrappers
apply experimental terms during Adam and final color polish.
Their coefficients and sampled weighted derivatives are saved separately;
total objectives with different terms are not cross-variant quality scores.

The historical `trace_solver.py` and `compare_discrete_layout.py` refinement
comparisons explicitly set `perceptual_loss=False`. Normal Python solves and
fixed-glyph color fits enable it by default; `solve_style` enables it only for
headroom style. Kernels, angle bins, source brightness weights, and edge weights
are cached per solve. The production loss requests only the selected metrics.

`view_renders.py` opens a local comparison tool for the saved experiments:

```sh
.venv/bin/python scripts/view_renders.py
# Or supply your own pair:
.venv/bin/python scripts/view_renders.py \
  --image 'Source=/path/source.png' --image 'Render=/path/render.png' --ppd 130
```

Open the printed localhost URL. Choose Vivian or Tree, then any source,
installed render, experiment, or optional Adam snapshot. Space alternates A/B;
Enter starts/stops timed flipping; W opens the wipe; E computes the official
FLIP error map; F toggles fullscreen. Wheel zoom and drag pan apply to both
images together. 0 fits; 1 maps one image pixel to one physical display pixel.
Both selected images decode before flipping, with no crossfade or reload on
each flip. Different image dimensions retain their pixel scale instead of
stretching; FLIP requires equal dimensions.

FLIP runs on demand on the CPU using `flip-evaluator` from `.[diagnostics]`,
and caches up to eight maps in memory. Its nominal pixels-per-degree setting
comes from the experiment report and is editable; zoom does not silently
change its evaluation geometry. The viewer serves registered files only,
binds to localhost, and does not rerender or update wallpaper settings/caches.

For the native experiment's 14×28 glyph cells, the contour windows are
56×56 and 112×112 image pixels: four columns by two rows and eight columns
by four rows, respectively, overlapping at half-window strides. Larger
windows could encourage consistency along longer contours, but combine
neighboring curves and lose positional information. An additional low-weight
224×224 scale would be a useful separate experiment; it has not been tested.

`compare_objective.py` traces the current production objective on any source,
optionally using an installed manifest for exact font, glyph and appearance
settings. It saves the source, final render, initialization, raw Adam snapshots,
terminal state, per-term derivatives and all candidate stroke costs. The
brightness reference is the source mapped into the atlas display range;
independent metrics continue to compare with the original source.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_objective.py image.jpg \
  --manifest /path/to/completed/manifest.json --label fix --output rendered/review
.venv/bin/python scripts/view_renders.py \
  --image Source=rendered/review/source.png \
  --image Current=rendered/review/current/final.png \
  --image Revised=rendered/review/fix/final.png
```

Capture `current` with the original revision before editing, or run it from
a separate checkout on `PYTHONPATH`. Use identical settings for both runs.
`stroke_costs.npz` records per-cell/per-glyph angular costs; `trace.json` records
the fixed linear-RGB reference range and effective coefficients, including the
0.1 pixel-family scale and 0.2 relative-chroma term. Source/image-specific crops belong in analysis only;
production does not inspect glyph names, semantic regions, or image identities.

Normal production headroom renders now return the raw Adam terminal. Use `compare_objective.py --postprocess` or the Python `postprocess=True` option to inspect optional dither and color polish. Existing comparison images retain their original meaning; changing the output stage invalidates render caches through the source revision.

## Animated flow review

`compare_flow_playback.py` prepares real production flow clips, reconstructs
previews from their serialized maps, and compares static flow, animation, and
stronger eddies on the four standard sources. It records ordinary and seam
pixel/lightness steps, temporal acceleration, glyph changes, field closure,
sampled official FLIP scores, runtime, and source/font/code provenance.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_flow_playback.py \
  --device cuda --width 768 --seconds 10 --fps 12
.venv/bin/python scripts/view_renders.py --report rendered/flow_playback/review
```

Use `--images tree` or `--variants flow_animated` for a smaller pilot, with a
separate `--output` directory to retain the complete review report. `--progress`
prints the production render stages; `--force` rebuilds matching outputs.

In the viewer, P plays/pauses, the slider selects a frame, and **Inspect loop
seam** repeats the last and first eight frames. Inspect the internal last/first
join; the inspection window itself has an artificial restart. FLIP pauses
playback and compares the selected frames. See [the implementation guide](../docs/flow-playback.md)
for the temporal overlap boundary and its physical limitations.

## Slideshow transition review

`compare_transitions.py` compares a completed slideshow's saved transitions with
current transitions using exactly the same saved key frames and glyph atlas.
It writes separate review assets, full-resolution lightness measurements, and
previews for both transition directions. It leaves the input directory intact.

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_transitions.py \
  /path/to/completed/assets --output rendered/transitions/review
.venv/bin/python scripts/view_renders.py --report rendered/transitions/review
```

Choose **original** for A and **oklab interpolation** for B. P plays the whole
slideshow with short holds at each key frame; Space switches between the
synchronized versions. The frame slider supports paused comparisons.

Add `--transition-type earth-mover` to compare spatial transport against the saved
transitions using the same stills and atlas, then select **earth mover** for B.
The report records the chosen type in the revised asset manifest.
