/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import "../../contents/ui/RenderSettings.js" as RenderSettings

TestCase {
    function test_flow_request_preserves_style_controls_and_variety() {
        const request = RenderSettings.request([], {RenderStyle: "flow", FlowStrength: 0.8,
            FlowEddies: 1.2, FlowScale: 4, GlyphEntropyStrength: 0.03})
        compare(request.style, "flow")
        compare(request.perceptual_loss, true)
        compare(request.flow_strength, 0.8)
        compare(request.flow_eddies, 1.2)
        compare(request.flow_scale, 4)
        compare(request.glyph_entropy_strength, 0.03)
    }
    name: "RenderSettings"

    function test_source_modes_and_video_options_use_shared_request_schema() {
        for (const mode of ["Image", "Slideshow", "Video", "Archive"]) {
            const config = RenderSettings.request(["/source"], {Mode: mode, FrameStride: 3,
                VideoTransitionFrames: 2, RenderStyle: "flow", FlowAnimated: true})
            const wallpaper = RenderSettings.request(["/source"], {mode: mode, frameStride: 3,
                videoTransitionFrames: 2, renderStyle: "flow", flowAnimated: true})
            compare(JSON.stringify(config), JSON.stringify(wallpaper))
            compare(config.input_format, mode === "Video" ? "video" : mode === "Archive" ? "archive" : "images")
            compare(config.frame_stride, 3)
            compare(config.video_transition_frames, 2)
            compare(config.flow_animated, mode === "Image" || mode === "Slideshow")
        }
        const defaults = RenderSettings.request([], {Mode: "Video"})
        compare(defaults.frame_stride, 1)
        compare(defaults.video_transition_frames, 0)
    }

    function test_defaults_match_service_and_cli() {
        const request = RenderSettings.request([], {})
        for (const key of Object.keys(request)) {
            if (key !== "images") compare(request[key], TestConfig.renderDefaults[key], key)
        }
    }

    function test_config_and_wallpaper_profiles_make_same_request() {
        const config = RenderSettings.request(["/image.png"], {FontFamily: "Custom",
            AsciiChars: [" ", "@"], Detail: 0, TransitionFrames: 3, BackgroundMode: 1,
            BackgroundColor: "#112233", InvertInk: true, Grayscale: true, GlyphEntropyStrength: 0.024})
        const wallpaper = RenderSettings.request(["/image.png"], {fontFamily: "Custom",
            characters: " @", detail: 0, transitionFrames: 3, backgroundMode: 1,
            backgroundColor: "#112233", invertInk: true, grayscale: true, glyphEntropyStrength: 0.024})
        compare(JSON.stringify(config), JSON.stringify(wallpaper))
        compare(config.glyph_entropy_strength, 0.024)
    }

    function test_structure_clears_unavailable_objectives_and_normalizes_space() {
        const request = RenderSettings.request([], {RenderStyle: "structure",
            AsciiChars: ["\\s", ".", "@"], GradientLossStrength: 1.5, GlyphLuminanceBonus: 0.12,
            GlyphEntropyStrength: 0.01})
        compare(request.chars, " .@")
        compare(request.gradient_loss_strength, 0)
        compare(request.glyph_luminance_bonus, 0)
        compare(request.glyph_entropy_strength, 0)
    }

    function test_zero_entropy_setting_is_preserved() {
        compare(RenderSettings.request([], {GlyphEntropyStrength: 0}).glyph_entropy_strength, 0)
        compare(RenderSettings.request([], {glyphEntropyStrength: 0}).glyph_entropy_strength, 0)
    }

    function test_render_styles_select_distinct_objectives_and_ignore_unused_settings() {
        const perceptual = RenderSettings.request([], {RenderStyle: "headroom", StructureDither: 0.12})
        const classic = RenderSettings.request([], {RenderStyle: "classic", StructureDither: 0})
        const structure = RenderSettings.request([], {RenderStyle: "structure", Dither: 0.05,
            GradientLossStrength: 4, GlyphLuminanceBonus: 0.5, GlyphEntropyStrength: 0.1})
        compare(perceptual.style, "headroom")
        compare(perceptual.perceptual_loss, true)
        compare(classic.style, "headroom")
        compare(classic.perceptual_loss, false)
        compare(perceptual.structure_dither, classic.structure_dither)
        compare(classic.gradient_loss_strength, 1.5)
        compare(structure.perceptual_loss, false)
        compare(structure.dither, 0)
        compare(structure.gradient_loss_strength, 0)
        compare(structure.glyph_luminance_bonus, 0)
        compare(structure.glyph_entropy_strength, 0)
    }
}
