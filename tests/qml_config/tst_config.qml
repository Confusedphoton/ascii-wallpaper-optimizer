/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import Wallpaper 1.0

TestCase {
    id: tests
    name: "WallpaperConfig"
    when: windowShown
    visible: true
    width: 900
    height: 650

    Component { id: configComponent; Config { width: 850 } }

    function page(properties) {
        const item = createTemporaryObject(configComponent, tests, properties || {})
        verify(item)
        tryCompare(item, "controlsReady", true)
        findChild(item, "sourceCheck").enabled = false
        findChild(item, "renderClient").serviceUrl = TestConfig.serviceUrl
        findChild(item, "cacheClient").serviceUrl = TestConfig.serviceUrl
        return item
    }

    function snapshot(item) {
        const properties = {}
        for (const key of Object.keys(item)) {
            if (key.startsWith("cfg_") && !key.endsWith("Changed"))
                properties[key] = key.indexOf("Images") >= 0 || key.indexOf("AsciiChars") >= 0
                    ? Array.from(item[key]) : item[key]
        }
        return properties
    }

    function test_video_source_controls_defaults_and_profile_persistence() {
        const item = page({cfg_ProfileSettingsInitialized: true})
        item.editProfile = 0
        const modes = findChild(item, "modeCombo")
        compare(modes.count, 4)
        modes.activated(2)
        compare(item.editMode, "Video")
        compare(item.activeSettings.FrameStride, 1)
        compare(item.activeSettings.VideoTransitionFrames, 0)
        const stride = findChild(item, "frameStrideSpin")
        const interpolation = findChild(item, "videoTransitionFramesSpin")
        verify(stride.visible)
        verify(interpolation.visible)
        compare(stride.from, 1)
        compare(stride.to, 10000)
        compare(interpolation.from, 0)
        compare(interpolation.to, 118)
        verify(!findChild(item, "transitionType").visible)
        item.setProfileSetting("Images", ["/one.mp4", "/ignored.mp4"])
        item.setProfileSetting("FrameStride", 3)
        item.setProfileSetting("VideoTransitionFrames", 2)
        item.setProfileSetting("TransitionType", "earth-mover")
        item.setProfileSetting("RenderStyle", "flow")
        item.setProfileSetting("FlowAnimated", true)
        verify(findChild(item, "transitionType").visible)
        verify(!findChild(item, "flowAnimatedCheck").visible)
        verify(!findChild(item, "flowLoopSecondsSpin").visible)
        const request = findChild(item, "renderClient").requestSettings
        compare(request.input_format, "video")
        compare(request.images, ["/one.mp4"])
        compare(request.frame_stride, 3)
        compare(request.video_transition_frames, 2)
        compare(request.transition_type, "earth-mover")
        compare(request.flow_animated, false)
        item.editProfile = 1
        item.setProfileSetting("Mode", "Video")
        item.setProfileSetting("FrameStride", 5)
        compare(item.activeSettings.VideoTransitionFrames, 0)
        item.saveConfig()
        const saved = snapshot(item)
        compare(saved.cfg_FrameStride, 3)
        compare(saved.cfg_VideoTransitionFrames, 2)
        compare(saved.cfg_FrameStrideB, 5)
        compare(saved.cfg_VideoTransitionFramesB, 0)
        item.destroy()
        const reopened = page(saved)
        reopened.editProfile = 0
        compare(reopened.activeSettings.FrameStride, 3)
        compare(reopened.activeSettings.VideoTransitionFrames, 2)
        reopened.editProfile = 1
        compare(reopened.activeSettings.FrameStride, 5)
        compare(reopened.activeSettings.VideoTransitionFrames, 0)
    }

    function test_archive_source_uses_single_bundle_and_hides_appearance() {
        const item = page({cfg_ProfileSettingsInitialized: true})
        item.editProfile = 0
        findChild(item, "modeCombo").activated(3)
        compare(item.editMode, "Archive")
        item.setProfileSetting("Images", ["/prepared.ascii.zip", "/ignored.zip"])
        const request = findChild(item, "renderClient").requestSettings
        compare(request.input_format, "archive")
        compare(request.images, ["/prepared.ascii.zip"])
        verify(findChild(item, "archiveSourceHint").visible)
        verify(!findChild(item, "renderStyleCombo").visible)
        verify(!findChild(item, "characterField").visible)
        verify(!findChild(item, "frameStrideSpin").visible)
        verify(!findChild(item, "videoTransitionFramesSpin").visible)
        verify(!findChild(item, "transitionType").visible)
        const client = findChild(item, "renderClient")
        client.displayPlaybackSettings = {mode: "Archive"}
        client.displayRequestSettings = client.requestSettings
        client.displayManifest = {image_count: 2}
        verify(findChild(item, "previewPlayback").slideshowEnabled)
        verify(item.showSlideshowTiming)
        client.displayManifest = {image_count: 1, video: {path: "video", frame_count: 3,
            duration_ms: 300, frame_durations_ms: [100, 100, 100]}}
        verify(!findChild(item, "previewPlayback").slideshowEnabled)
        verify(!item.showSlideshowTiming)
        item.setProfileSetting("Mode", "Image")
        verify(findChild(item, "renderStyleCombo").visible)
        verify(findChild(item, "characterField").visible)
    }

    function test_earth_mover_selection_persists_across_profiles() {
        const item = page({cfg_Mode: "Slideshow", cfg_ModeB: "Slideshow"})
        item.editProfile = 0
        const selector = findChild(item, "transitionType")
        verify(selector.visible)
        compare(selector.currentIndex, 0)
        selector.activated(1)
        compare(item.activeSettings.TransitionType, "earth-mover")
        compare(findChild(item, "renderClient").requestSettings.transition_type, "earth-mover")
        item.editProfile = 1
        compare(item.activeSettings.TransitionType, "glyph-path")
        item.saveConfig()
        const saved = snapshot(item)
        item.destroy()
        const reopened = page(saved)
        reopened.editProfile = 0
        compare(reopened.activeSettings.TransitionType, "earth-mover")
        compare(findChild(reopened, "transitionType").currentIndex, 1)
        compare(findChild(reopened, "renderClient").requestSettings.transition_type, "earth-mover")
        reopened.editProfile = 1
        compare(reopened.activeSettings.TransitionType, "glyph-path")
    }

    function test_flow_animation_options_persist_across_profiles() {
        const item = page({cfg_ProfileSettingsInitialized: true})
        item.editProfile = 0
        item.setProfileSetting("RenderStyle", "flow")
        item.setProfileSetting("FlowAnimated", true)
        item.setProfileSetting("FlowLoopSeconds", 6)
        item.setProfileSetting("FlowFps", 16)
        item.editProfile = 1
        compare(item.activeSettings.FlowAnimated, false)
        item.saveConfig()
        const saved = snapshot(item)
        item.destroy()
        const reopened = page(saved)
        reopened.editProfile = 0
        compare(reopened.activeSettings.FlowAnimated, true)
        compare(reopened.activeSettings.FlowLoopSeconds, 6)
        compare(reopened.activeSettings.FlowFps, 16)
        verify(findChild(reopened, "flowAnimatedCheck").visible)
        reopened.editProfile = 1
        compare(reopened.activeSettings.FlowAnimated, false)
    }

    function test_saved_sources_survive_profile_switch_apply_and_reopen() {
        const a = ["/pictures/A%20literal.png", "/pictures/A,2.png"]
        const b = ["/pictures/日本語.png", "/pictures/B%2F.png"]
        const item = page({cfg_Mode: "Slideshow", cfg_ModeB: "Slideshow",
            cfg_Images: a, cfg_ImagesB: b, cfg_ManualProfile: "B"})
        item.editProfile = 0
        compare(Array.from(item.editImages), a)
        item.editProfile = 1
        compare(Array.from(item.editImages), b)
        const list = findChild(item, "imageList")
        list.append({path: "/pictures/B 3.png"})
        item.syncToConfig()
        item.saveConfig()
        const saved = snapshot(item)
        item.destroy()
        wait(0)
        const reopened = page(saved)
        compare(reopened.editProfile, 1)
        compare(reopened.editMode, "Slideshow")
        compare(Array.from(reopened.cfg_Images), a)
        compare(Array.from(reopened.cfg_ImagesB), b.concat(["/pictures/B 3.png"]))
        compare(findChild(reopened, "imageList").count, 3)
        reopened.saveConfig()
        compare(Array.from(reopened.cfg_ImagesB), b.concat(["/pictures/B 3.png"]))
    }

    function test_last_editing_profile_persists_without_applying_wallpaper() {
        const item = page({cfg_ManualProfile: "A"})
        item.editProfile = 0
        item.editProfile = 1
        item.destroy()
        wait(0)
        const reopened = page({cfg_ManualProfile: "A"})
        compare(reopened.editProfile, 1)
        compare(reopened.cfg_ManualProfile, "A")
    }

    function test_migration_waits_for_initial_properties() {
        const item = page({cfg_Image: "file:///pictures/legacy%20image.png",
            cfg_Mode: "Slideshow", cfg_Images: ["/pictures/saved.png"],
            cfg_ImagesB: ["/pictures/b.png"], cfg_ManualProfile: "A"})
        item.editProfile = 0
        compare(Array.from(item.cfg_Images), ["/pictures/saved.png", "/pictures/legacy image.png"])
        compare(Array.from(item.cfg_ImagesB), ["/pictures/b.png"])
    }

    function test_character_text_preserves_spaces_unicode_and_profiles() {
        const item = page({cfg_ProfileSettingsInitialized: true,
            cfg_AsciiChars: [" ", ".", "@"], cfg_AsciiCharsB: [" ", "#"]})
        item.editProfile = 0
        verify(item.setCharacters(" .é😀#"))
        compare(Array.from(item.cfg_AsciiChars).join(""), " .é😀#")
        verify(!item.setCharacters(" .."))
        verify(!item.setCharacters(""))
        verify(!item.setCharacters("a\tb"))
        compare(Array.from(item.cfg_AsciiChars).join(""), " .é😀#")
        item.editProfile = 1
        compare(findChild(item, "characterField").text, " #")
        const field = findChild(item, "characterField")
        field.text = " #é"
        item.saveConfig()
        compare(Array.from(item.cfg_AsciiCharsB).join(""), " #é")
        item.editProfile = 0
        compare(findChild(item, "characterField").text, " .é😀#")
    }

    function test_apply_submits_changed_settings() {
        const item = page({cfg_Images: ["/apply.png"], cfg_ProfileSettingsInitialized: true})
        item.editProfile = 0
        const client = findChild(item, "renderClient")
        item.setProfileSetting("BackgroundMode", 1)
        item.setProfileSetting("BackgroundColor", "#112233")
        wait(50)
        verify(!client.submissionPending)
        verify(!client.requestInFlight)
        item.saveConfig()
        tryCompare(client, "displayGeneration", "#112233", 3000)
        verify(!client.currentRequestForced)
        verify(!client.settingsPending)
        item.setProfileSetting("BackgroundColor", "#445566")
        const previewLabel = findChild(item, "previewLabel")
        compare(previewLabel.text, "Last render")
        item.setProfileSetting("BackgroundColor", "#112233")
        tryCompare(previewLabel, "text", "Preview")
        item.setProfileSetting("BackgroundColor", "#445566")
        item.saveConfig()
        tryCompare(client, "displayGeneration", "#445566", 3000)
        verify(findChild(item, "previewShader").visible)
    }

    function test_preview_survives_render_and_failure() {
        const item = page({cfg_Images: ["/preview.png"]})
        const client = findChild(item, "renderClient")
        client.consume({state: "ready", directory: TestConfig.fixtureRoot, generation: "old",
            image_count: 1, frame_count: 4})
        const preview = findChild(item, "previewShader")
        tryCompare(preview, "ready", true, 3000)
        client.consume({state: "running", stage: "Rendering", progress: 0.2})
        compare(preview.displayGeneration, "old")
        compare(findChild(item, "previewLabel").text, "Last render")
        client.consume({state: "error", error: "Render failed"})
        compare(preview.displayGeneration, "old")
        verify(preview.visible)
        client.consume({state: "ready", directory: TestConfig.fixtureRoot, generation: "new",
            image_count: 1, frame_count: 4})
        tryCompare(preview, "displayGeneration", "new", 3000)
    }

    function test_source_choices_survive_mode_switch_and_reopen() {
        const images = ["/first.jpg", "/second.jpg"]
        const item = page({cfg_Mode: "Slideshow", cfg_Images: images})
        item.editProfile = 0
        const modes = findChild(item, "modeCombo")
        modes.activated(2)
        compare(Array.from(item.editImages), [])
        item.setProfileSetting("Images", ["/clip.mkv"])
        modes.activated(1)
        compare(Array.from(item.editImages), images)
        const saved = snapshot(item)
        item.destroy()
        const reopened = page(saved)
        reopened.editProfile = 0
        findChild(reopened, "modeCombo").activated(2)
        compare(Array.from(reopened.editImages), ["/clip.mkv"])
    }

    function test_custom_23_glyphs_survive_opening_editor() {
        const chars = "abcdefghijklmnopqrstuvw"
        const item = page({cfg_Profiles: JSON.stringify([
            {id: "A", name: "Profile A", startTime: "07:00", settings: {AsciiChars: Array.from(chars)}}
        ])})
        compare(findChild(item, "characterField").text, chars)
        compare(findChild(item, "renderClient").requestSettings.chars, chars)
    }

    function test_named_profiles_persist_sources_settings_and_schedule() {
        const item = page()
        item.addProfile()
        item.setProfileDetails("name", "Afternoon")
        verify(item.setProfileDetails("startTime", "13:30"))
        verify(!item.setProfileDetails("startTime", "07:00"))
        verify(!item.setProfileDetails("startTime", "25:00"))
        item.setProfileSetting("Images", ["/afternoon.png"])
        item.setProfileSetting("RenderStyle", "classic")
        item.setProfileSetting("Detail", 4)
        const id = item.profiles[item.editProfile].id
        item.cfg_ManualProfile = id
        item.cfg_ScheduleEnabled = true
        item.addProfile()
        verify(item.setProfileDetails("name", "Evening"))
        verify(item.setProfileDetails("startTime", "20:00"))
        item.editProfile = 2
        item.saveConfig()
        const saved = snapshot(item)
        item.destroy()
        wait(0)
        const reopened = page(saved)
        compare(reopened.profiles.length, 4)
        compare(reopened.editProfile, 2)
        compare(reopened.editingProfile.name, "Afternoon")
        compare(reopened.editingProfile.startTime, "13:30")
        compare(Array.from(reopened.editImages), ["/afternoon.png"])
        compare(reopened.activeSettings.RenderStyle, "classic")
        compare(reopened.activeSettings.Detail, 4)
        compare(reopened.cfg_ManualProfile, id)
        compare(findChild(reopened, "editingProfileCombo").count, 4)
        compare(findChild(reopened, "manualProfileCombo").count, 4)
        reopened.removeProfile()
        compare(reopened.profiles.length, 3)
        compare(reopened.cfg_ManualProfile, "A")
        while (reopened.profiles.length > 1) reopened.removeProfile()
        reopened.removeProfile()
        compare(reopened.profiles.length, 1)
    }

    function test_each_style_shows_all_and_only_its_controls() {
        const item = page({cfg_PerceptualLoss: false})
        item.editProfile = 0
        compare(item.activeSettings.RenderStyle, "classic")
        const headroom = ["gradientLossSlider", "glyphLuminanceBonusSlider", "glyphEntropySlider"]
        for (const style of ["classic", "headroom", "structure", "flow"]) {
            item.setProfileSetting("RenderStyle", style)
            for (const name of headroom) compare(findChild(item, name).visible, style !== "structure")
            compare(findChild(item, "ditherSlider").visible, style === "classic")
            compare(findChild(item, "structureDitherSlider").visible, style === "structure")
            for (const name of ["flowStrengthSlider", "flowEddiesSlider", "flowScaleSlider"])
                compare(findChild(item, name).visible, style === "flow")
            const request = findChild(item, "renderClient").requestSettings
            compare(request.perceptual_loss, style === "headroom" || style === "flow")
        }
        compare(findChild(item, "renderStyleCombo").count, 4)
    }

    function test_flow_bank_controls_and_custom_characters_persist() {
        const item = page({cfg_RenderStyle: "flow"})
        item.editProfile = 0
        item.setProfileSetting("RenderStyle", "flow")
        compare(item.activeSettings.AsciiChars.length, 96)
        item.setProfileSetting("FlowStrength", 0.4)
        item.setProfileSetting("FlowEddies", 1.2)
        item.setProfileSetting("FlowScale", 4)
        item.saveConfig()
        const restored = page(snapshot(item))
        restored.editProfile = 0
        compare(restored.activeSettings.RenderStyle, "flow")
        compare(restored.activeSettings.AsciiChars.length, 96)
        compare(restored.activeSettings.FlowStrength, 0.4)
        compare(restored.activeSettings.FlowEddies, 1.2)
        compare(restored.activeSettings.FlowScale, 4)
        restored.setCharacters(" .G@")
        restored.setProfileSetting("RenderStyle", "headroom")
        restored.applyStyleCharacters("headroom")
        compare(restored.activeSettings.AsciiChars.join(""), " .G@")
    }

    function test_cache_limit_and_clear_are_connected() {
        const item = page({cfg_MaxCachedRenders: 7})
        item.editProfile = 0
        compare(findChild(item, "maxCachedRendersSpin").value, 7)
        item.saveConfig()
        const cache = findChild(item, "cacheClient")
        tryCompare(cache, "busy", false, 3000)
        findChild(item, "clearCacheButton").clicked()
        tryCompare(cache, "busy", false, 3000)
        verify(cache.message.indexOf("Cache cleared") >= 0)
    }

    function test_completed_preview_contains_visible_color() {
        if (GraphicsInfo.api === GraphicsInfo.Software) skip("RHI is required")
        const item = page({cfg_ProfileSettingsInitialized: true})
        const client = findChild(item, "renderClient")
        client.physicalWidth = 64
        client.physicalHeight = 64
        client.consume({state: "ready", directory: TestConfig.fixtureRoot,
            generation: "preview", image_count: 1, frame_count: 4,
            manifest: {output_size: [4, 4], cell_size: [2, 2], atlas_count: 1}})
        const preview = findChild(item, "previewShader")
        tryCompare(preview, "ready", true, 3000)
        tryCompare(preview, "loading", false, 3000)
        wait(100)
        const pixels = grabImage(preview)
        const x = Math.floor(pixels.width / 2), y = Math.floor(pixels.height / 2)
        verify(pixels.red(x, y) > 100, "Completed preview must contain the rendered image")
        verify(pixels.green(x, y) > 50)
        verify(pixels.red(x, y) > pixels.blue(x, y))
        compare(pixels.alpha(x, y), 255)
        compare(JSON.parse(item.cfg_RenderedState).generation, "preview")
        const point = preview.mapToItem(item, preview.width / 2, preview.height / 2)
        const pagePixels = grabImage(item)
        verify(pagePixels.red(Math.floor(point.x), Math.floor(point.y)) > 100,
            "The preview must be visible in the actual config page")
        pagePixels.save(TestConfig.fixtureRoot + "/settings.png")
        const reopened = page(snapshot(item))
        const restored = findChild(reopened, "renderClient")
        compare(restored.displayGeneration, "preview")
        compare(restored.displayManifest.atlas_count, 1)
    }

    function test_glyph_variety_slider_persists_each_profile_and_zero() {
        const item = page({cfg_ProfileSettingsInitialized: true})
        item.editProfile = 0
        const slider = findChild(item, "glyphEntropySlider")
        compare(slider.value, 0.01)
        compare(slider.from, 0)
        compare(slider.to, 0.1)
        slider.forceActiveFocus()
        for (let step = 0; step < 15; step++) keyClick(Qt.Key_Right)
        compare(item.cfg_GlyphEntropyStrength, 0.025)
        item.editProfile = 1
        compare(slider.value, 0.01)
        slider.forceActiveFocus()
        for (let step = 0; step < 10; step++) keyClick(Qt.Key_Left)
        compare(item.cfg_GlyphEntropyStrengthB, 0)
        item.editProfile = 0
        compare(slider.value, 0.025)
        item.saveConfig()
        const reopened = page(snapshot(item))
        reopened.editProfile = 0
        compare(findChild(reopened, "glyphEntropySlider").value, 0.025)
        reopened.editProfile = 1
        compare(findChild(reopened, "glyphEntropySlider").value, 0)
        reopened.setProfileSetting("RenderStyle", "structure")
        verify(!findChild(reopened, "glyphEntropySlider").visible)
    }
}
