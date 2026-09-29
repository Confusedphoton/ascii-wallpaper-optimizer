/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import "../../contents/ui/RenderSettings.js" as RenderSettings

TestCase {
    name: "RenderSettings"

    function test_defaults_match_service_and_cli() {
        const request = RenderSettings.request([], {})
        for (const key of Object.keys(request)) {
            if (key !== "images") compare(request[key], TestConfig.renderDefaults[key], key)
        }
    }

    function test_config_and_wallpaper_profiles_make_same_request() {
        const config = RenderSettings.request(["/image.png"], {FontFamily: "Custom",
            AsciiChars: [" ", "@"], Detail: 0, TransitionFrames: 3, BackgroundMode: 1,
            BackgroundColor: "#112233", InvertInk: true, Grayscale: true})
        const wallpaper = RenderSettings.request(["/image.png"], {fontFamily: "Custom",
            characters: " @", detail: 0, transitionFrames: 3, backgroundMode: 1,
            backgroundColor: "#112233", invertInk: true, grayscale: true})
        compare(JSON.stringify(config), JSON.stringify(wallpaper))
    }

    function test_structure_clears_unavailable_objectives_and_normalizes_space() {
        const request = RenderSettings.request([], {RenderStyle: "structure",
            AsciiChars: ["\\s", ".", "@"], GradientLossStrength: 1.5, GlyphLuminanceBonus: 0.12})
        compare(request.chars, " .@")
        compare(request.gradient_loss_strength, 0)
        compare(request.glyph_luminance_bonus, 0)
    }
}
