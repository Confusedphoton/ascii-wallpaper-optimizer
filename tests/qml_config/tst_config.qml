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
        const reopened = page(snapshot(item))
        const restored = findChild(reopened, "renderClient")
        compare(restored.displayGeneration, "preview")
        compare(restored.displayManifest.atlas_count, 1)
        const point = preview.mapToItem(item, preview.width / 2, preview.height / 2)
        const pagePixels = grabImage(item)
        verify(pagePixels.red(Math.floor(point.x), Math.floor(point.y)) > 100,
            "The preview must be visible in the actual config page")
        pagePixels.save(TestConfig.fixtureRoot + "/settings.png")
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
