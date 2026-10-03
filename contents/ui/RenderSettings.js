/* SPDX-License-Identifier: AGPL-3.0-or-later */
.pragma library

// Settings UI stores KConfig names; wallpaper resolves profiles to camel case.
function request(images, settings) {
    function setting(camel, config, fallback) {
        if (settings[camel] !== undefined) return settings[camel]
        if (settings[config] !== undefined) return settings[config]
        return fallback
    }
    const style = setting("renderStyle", "RenderStyle", "headroom")
    const animated = style === "flow" && !!setting("flowAnimated", "FlowAnimated", false)
    const headroom = style !== "structure"
    const characters = setting("characters", "AsciiChars", " .´:rszunvcxLOGDQ0KNMW&#\\/=^|[]{}")
    return {
        images: Array.from(images || []),
        font: setting("fontFamily", "FontFamily", "Monospace"),
        chars: Array.from(characters).map((glyph, index) => index === 0 && glyph === "\\s" ? " " : glyph).join(""),
        detail: setting("detail", "Detail", 1),
        grayscale: !!setting("grayscale", "Grayscale", false),
        frames: setting("transitionFrames", "TransitionFrames", 16),
        transition_pace: setting("transitionPace", "TransitionPace", "ease-in-out"),
        transition_field: setting("transitionField", "TransitionField", "blue-noise"),
        transition_spread: setting("transitionSpread", "TransitionSpread", 0.65),
        dither: headroom ? setting("dither", "Dither", 0.005) : 0,
        glyph_luminance_bonus: headroom ? setting("glyphLuminanceBonus", "GlyphLuminanceBonus", 0.12) : 0,
        glyph_entropy_strength: headroom ? setting("glyphEntropyStrength", "GlyphEntropyStrength", style === "flow" ? 0.02 : 0.01) : 0,
        gradient_loss_strength: headroom ? setting("gradientLossStrength", "GradientLossStrength", 1.5) : 0,
        structure_dither: style === "structure" ? setting("structureDither", "StructureDither", 0.06) : 0.06,
        flow_strength: setting("flowStrength", "FlowStrength", 0.4),
        flow_eddies: setting("flowEddies", "FlowEddies", 0.65),
        flow_scale: setting("flowScale", "FlowScale", 6.0),
        flow_animated: animated,
        flow_loop_seconds: animated ? setting("flowLoopSeconds", "FlowLoopSeconds", 10) : 10,
        flow_fps: animated ? setting("flowFps", "FlowFps", 12) : 12,
        flow_frame_steps: 12,
        flow_temporal_strength: 0.8,
        style: style === "flow" ? "flow" : headroom ? "headroom" : "structure",
        perceptual_loss: style === "headroom" || style === "flow",
        invert_ink: !!setting("invertInk", "InvertInk", false),
        background: setting("backgroundMode", "BackgroundMode", 0) === 1
            ? setting("backgroundColor", "BackgroundColor", "#202028") : "auto"
    }
}
