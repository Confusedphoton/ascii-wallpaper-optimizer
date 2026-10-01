"""Comparison helpers using the production spatial metrics."""

from ascii_optimizer import optimizer
from ascii_optimizer.perceptual import (  # noqa: F401
    SpatialMetrics,
    blur,
    circular_w1,
    gradients,
    lightness,
    luminance,
)


class ExperimentalObjective:
    """Script-only wrapper, used with refinement disabled (no exact delta cache)."""

    def __init__(self, baseline, metrics, weights, replace_sobel=False):
        self.baseline, self.metrics, self.weights = baseline, metrics, weights
        if replace_sobel:
            baseline.gradient_strength = 0

    def __getattr__(self, name):
        return getattr(self.baseline, name)

    def __call__(self, rendered, *, components=None, displayed_lab=None):
        displayed = (
            displayed_lab
            if displayed_lab is not None
            else optimizer._display_headroom_lab(
                rendered, invert_ink=self.baseline.invert_ink
            )
        )
        value = self.baseline(rendered, components=components, displayed_lab=displayed)
        rgb = optimizer._oklab_to_rgb(displayed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        terms, _ = self.metrics.terms(
            rgb,
            orientation=any(k.startswith("orientation") for k in self.weights),
            stroke="orientation_stroke_tensor" in self.weights,
        )
        for name, weight in self.weights.items():
            term = weight * terms[name]
            value = value + term
            if components is not None:
                components[name] = term
        return value
