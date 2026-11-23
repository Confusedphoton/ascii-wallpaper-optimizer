/* SPDX-License-Identifier: AGPL-3.0-or-later */
.pragma library
.import "Profiles.js" as Profiles

// Settings UI stores KConfig names; wallpaper resolves profiles to camel case.
function request(images, settings) {
    const d = Profiles.defaults
    function setting(camel, config, fallback) {
        if (settings[camel] !== undefined) return settings[camel]
        if (settings[config] !== undefined) return settings[config]
        return fallback !== undefined ? fallback : d[config]
    }
    const style = setting("renderStyle", "RenderStyle")
    const mode = setting("mode", "Mode")
    const inputFormat = mode === "Video" ? "video" : mode === "Archive" ? "archive" : "images"
    const animated = inputFormat === "images" && style === "flow"
        && !!setting("flowAnimated", "FlowAnimated")
    const stroke = style === "stroke"
    const characters = setting("characters", "AsciiChars", Profiles.defaultCharsForStyle(style))
    return {
        images: Array.from(images || []),
        input_format: inputFormat,
        frame_stride: setting("frameStride", "FrameStride"),
        video_transition_frames: setting("videoTransitionFrames", "VideoTransitionFrames"),
        font: setting("fontFamily", "FontFamily"),
        chars: Array.from(characters).map((glyph, index) => index === 0 && glyph === "\\s" ? " " : glyph).join(""),
        detail: setting("detail", "Detail"),
        grayscale: !!setting("grayscale", "Grayscale"),
        frames: setting("transitionFrames", "TransitionFrames"),
        transition_type: setting("transitionType", "TransitionType"),
        transition_pace: setting("transitionPace", "TransitionPace"),
        transition_field: setting("transitionField", "TransitionField"),
        transition_spread: setting("transitionSpread", "TransitionSpread"),
        dither: stroke ? 0 : setting("dither", "Dither"),
        glyph_luminance_bonus: stroke ? 0 : setting("glyphLuminanceBonus", "GlyphLuminanceBonus"),
        glyph_entropy_strength: stroke ? 0 : setting("glyphEntropyStrength", "GlyphEntropyStrength",
            style === "flow" ? 0.02 : d.GlyphEntropyStrength),
        gradient_loss_strength: stroke ? 0 : setting("gradientLossStrength", "GradientLossStrength"),
        flow_strength: setting("flowStrength", "FlowStrength"),
        flow_eddies: setting("flowEddies", "FlowEddies"),
        flow_scale: setting("flowScale", "FlowScale"),
        flow_animated: animated,
        flow_loop_seconds: animated ? setting("flowLoopSeconds", "FlowLoopSeconds") : d.FlowLoopSeconds,
        flow_fps: animated ? setting("flowFps", "FlowFps") : d.FlowFps,
        flow_frame_steps: 12,
        flow_temporal_strength: 0.8,
        style: style === "flow" ? "flow" : stroke ? "stroke" : "headroom",
        perceptual_loss: style === "headroom" || style === "flow",
        invert_ink: !!setting("invertInk", "InvertInk"),
        background: setting("backgroundMode", "BackgroundMode") === 1
            ? setting("backgroundColor", "BackgroundColor") : "auto"
    }
}
