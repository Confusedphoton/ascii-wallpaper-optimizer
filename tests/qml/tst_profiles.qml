/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import "../../contents/ui/Profiles.js" as Profiles

TestCase {
    name: "Profiles"

    function test_flow_default_bank_and_controls_survive_profile_roundtrip() {
        const settings = Profiles.normalizedSettings({RenderStyle: "flow", FlowEddies: 1.2})
        compare(settings.AsciiChars.length, 96)
        verify(settings.AsciiChars.includes("@"))
        compare(new Set(settings.AsciiChars).size, 96)
        const profile = {id: "flow", name: "Flow", startTime: "12:00", settings: settings}
        const restored = Profiles.decode(JSON.stringify([profile]))[0].settings
        compare(restored.AsciiChars.join(""), Profiles.flowChars)
        compare(restored.FlowEddies, 1.2)
        compare(Profiles.normalizedSettings({RenderStyle: "flow", AsciiChars: [" ", "G"]}).AsciiChars.join(""), " G")
    }

    function test_video_options_default_and_migrate_independent_profiles() {
        const defaults = Profiles.normalizedSettings({})
        compare(defaults.FrameStride, 1)
        compare(defaults.VideoTransitionFrames, 0)
        const profiles = Profiles.fromLegacy({ProfileSettingsInitialized: true,
            Mode: "Video", Images: ["/one.mp4"], FrameStride: 4, VideoTransitionFrames: 2,
            ModeB: "Video", ImagesB: ["/two.mp4"], FrameStrideB: 7, VideoTransitionFramesB: 0})
        const restored = Profiles.decode(JSON.stringify(profiles))
        compare(restored[0].settings.Mode, "Video")
        compare(restored[0].settings.FrameStride, 4)
        compare(restored[0].settings.VideoTransitionFrames, 2)
        compare(restored[1].settings.FrameStride, 7)
        compare(restored[1].settings.VideoTransitionFrames, 0)
    }

    function test_multiple_start_times_wrap_midnight_and_ignore_list_order() {
        const profiles = [{id: "night", startTime: "22:00"}, {id: "day", startTime: "07:00"},
            {id: "afternoon", startTime: "13:30"}, {id: "evening", startTime: "18:00"}]
        const at = (hour, minute) => Profiles.selected(profiles, "day", true, new Date(2026, 0, 1, hour, minute))
        compare(at(0, 0), "night")
        compare(at(6, 59), "night")
        compare(at(7, 0), "day")
        compare(at(13, 29), "day")
        compare(at(13, 30), "afternoon")
        compare(at(18, 0), "evening")
        compare(at(22, 0), "night")
        compare(Profiles.selected(profiles, "afternoon", false, new Date()), "afternoon")
        compare(Profiles.selected(profiles, "removed", false, new Date()), "night")
    }

    function test_legacy_objective_and_sources_migrate_without_losing_settings() {
        const profiles = Profiles.fromLegacy({ProfileSettingsInitialized: true,
            Mode: "Slideshow", Images: ["/one.png", "/two.png"], Detail: 4, PerceptualLoss: false,
            RenderDirectory: "/saved", RenderedState: "saved-state", ProfileAStart: "06:00",
            ModeB: "Image", ImagesB: ["/three.png"], DetailB: 2, PerceptualLossB: true})
        compare(profiles[0].settings.RenderStyle, "classic")
        compare(profiles[1].settings.RenderStyle, "headroom")
        compare(profiles[0].settings.Images.length, 2)
        compare(profiles[0].settings.Detail, 4)
        compare(profiles[0].settings.RenderDirectory, "/saved")
        compare(profiles[0].settings.RenderedState, "saved-state")
        compare(profiles[0].startTime, "06:00")
        compare(profiles[1].settings.Detail, 2)
        compare(profiles[1].settings.Images[0], "/three.png")
        compare(Profiles.decode(JSON.stringify(profiles))[0].settings.RenderStyle, "classic")
    }

    function test_single_profile_and_invalid_storage_have_safe_defaults() {
        const profile = [{id: "only", name: "Only", startTime: "12:00", settings: {}}]
        compare(Profiles.selected(profile, "only", true, new Date(2026, 0, 1, 2, 0)), "only")
        compare(Profiles.read("broken", {}).length, 2)
        compare(Profiles.read("{}", {}).length, 2)
        compare(Profiles.read('[{"id":"auto"}]', {}).length, 2)
    }
}
