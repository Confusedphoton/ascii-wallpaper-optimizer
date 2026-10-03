# Animated forced-flow playback

Flow domains can prepare a looping clip for each source image. Enable **Animate flow** in wallpaper settings, or use the CLI:

```sh
.venv/bin/ascii-wallpaper-render --style flow --flow-animated \
  --flow-loop-seconds 10 --flow-fps 12 --output rendered/my-flow image.jpg
```

Animation remains disabled for existing profiles. The default clip is 120 frames over ten seconds. Duration accepts 1–30 seconds and frame rate accepts 4–24 fps. `--flow-frame-steps` sets the color-fitting budget per pass (default 12); `--flow-temporal-strength` controls cyclic glyph and color continuity (default 0.8, range 0–2). The existing `--frames` setting controls slideshow transitions independently.

## Closing the temporal boundary

The simulation retains its complete padded velocity field while advancing. Cropping occurs only when extracting a glyph-guidance frame. Its spatial domain is periodic, as in the static renderer. Static rendering retains its previous seeded trajectory and update order.

Animation warms the field for 96 steps, then records a continuous trajectory. Two seeded curl forces vary on independent time scales alongside the steady image-guided force. Their periods do not depend on the requested clip duration. The implementation assumes neither a converged periodic orbit nor equality between forcing and observed periods.

Each clip spans 48 simulation time units, sampled with timesteps no greater than 0.4. More output frames sample that interval more densely. Changing the displayed duration also changes playback speed; it does not prescribe a physical forcing period.

The renderer searches the recorded trajectory for a window with a relatively small endpoint state and step mismatch. It then imposes a temporal boundary on the rendered guidance field: the first quarter of the window blends the continued tail into the head with a quintic weight. Blending occurs across the complete padded field before projection and cropping. At phase zero the weight selects the endpoint continuation; at the end of the overlap it selects the unmodified head. The quintic weight has zero first and second derivatives at its endpoints. Exported phases cover `[0, T)` without duplicating the endpoint.

This produces an artistically closed guidance sequence, rather than an exact time-periodic solution of the fluid equations. Projection preserves spatial incompressibility, but the overlap correction can be substantial. `clip.json` records the raw state mismatch, correction magnitude, divergence, ordinary step sizes, and seam ratios. A nonfinite field or a velocity seam more than 2.5 times the mean step aborts generation and leaves the displayed generation available. Visual review remains necessary: that field check cannot guarantee that a discrete glyph switch is unobtrusive.

## Fitting a circular sequence

Source-image features and atlas descriptors are prepared once. Recorded simulation states and full-bank candidate costs use temporary disk-backed arrays; only changing field descriptors are recomputed. Every glyph remains available in every cell.

Glyph layouts minimize a local source/flow objective plus circular neighbor penalties based on actual atlas raster distances and frame-wide glyph entropy. Alternating sweeps include the last/first pair and accept only improvements to that objective. This layout objective is an approximation to the full perceptual image objective; it does not claim globally optimal glyph layouts.

Colors are then fitted with glyphs fixed using the existing full-frame perceptual objective. Both temporal neighbors constrain displayed cell tone, including across the wrap. Two opposing passes use equal budgets and analytic source-referenced starts, avoiding accumulated color drift around the cycle. The background remains fixed. Saved RGB8 maps are the playback result and the input to review metrics.

## Assets and playback

A schema-2 manifest contains an independent loop descriptor for each image, with its path, frame count, duration, anchor, and closure diagnostics. Each phase contains the usual glyph, color, and background maps. The still maps are exact copies of phase zero, so existing slideshow transitions connect the loop anchors.

The service validates every required frame before publishing an immutable generation. Loop settings, sources, font, and renderer revision participate in cache identity. Cache clearing and pruning include reusable clips as well as stills. Legacy manifests continue to play their existing static assets.

The QML loop clock runs for a single-image wallpaper as well as a slideshow. Slideshow dwell time is a minimum: playback completes a loop and reaches phase zero before starting the transition. A requested frame must load completely before replacing the visible output. Loading extends the playback clock, and a long UI stall does not trigger a burst of overdue frames. Hidden previews pause. A failed frame preserves the visible output and reports the error.

Playback performs map decoding and shader composition, with no simulation or optimizer in Plasma. The output texture is refreshed once per presented frame. Animation therefore adds recurring upload/composition work compared with a static wallpaper; the existing buffers and captured output texture remain bounded.

## Review and verification

```sh
.venv/bin/python -m ascii_optimizer.launcher scripts/compare_flow_playback.py \
  --device cuda --width 768 --seconds 10 --fps 12
.venv/bin/python scripts/view_renders.py --report rendered/flow_playback/review
```

The report compares static flow, animation, and stronger eddies on four source images. It reconstructs previews from serialized maps and records glyph changes, pixel/lightness steps, seam ratios, temporal acceleration, sampled official FLIP scores, runtime, and provenance.

In the viewer, **P** plays or pauses, the frame slider selects a phase, and **Inspect loop seam** repeats the last and first eight frames. The jump from the end of that inspection window back to its beginning is artificial; inspect the internal last-to-first boundary. **FLIP error** pauses playback and compares the selected frames. Space switches the A/B view.

Tests cover seeded stateful stepping, static parity, field closure and divergence, cached atlas costs, circular raster fitting, settings and asset validation, anchor identity, cache reuse, playback timing and failures, and actual shader output for both ink polarities across repeated wraps. Temporal overlap and discrete glyph motion should still be reviewed on the intended source, font, and display.
