/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import Wallpaper 1.0

TestCase {
    id: tests
    name: "ComposedFrame"
    when: windowShown
    visible: true
    width: 64
    height: 32
    Rectangle { anchors.fill: parent; color: "black"; z: -1 }

    Component { id: shaderComponent; AsciiShader { width: 16; height: 8 } }
    Component { id: clientComponent; RenderClient { serviceUrl: "http://127.0.0.1:1" } }
    Component { id: spyComponent; SignalSpy { signalName: "scheduledUpdateCompleted" } }
    Component {
        id: neighborComponent
        Rectangle {
            x: 32; width: 8; height: 8; color: "red"
            NumberAnimation on y { from: 0; to: 16; duration: 200; loops: Animation.Infinite }
        }
    }

    function initTestCase() {
        if (GraphicsInfo.api === GraphicsInfo.Software)
            skip("The Qt Quick RHI backend is unavailable")
    }

    function shader() {
        const item = createTemporaryObject(shaderComponent, tests, {
            renderDirectory: TestConfig.fixtureRoot, generation: "initial",
            manifest: TestConfig.manifest, imageCount: 2
        })
        tryCompare(item, "ready", true, 2000)
        tryCompare(item, "loading", false, 2000)
        return item
    }

    function checkPixels(item, path, scale) {
        const image = grabImage(item)
        const expected = TestConfig.pixels[path]
        scale *= image.width / item.width
        compare(image.width, 16 * scale)
        compare(image.height, 8 * scale)
        for (let y = 0; y < 8 * scale; y++) {
            for (let x = 0; x < 16 * scale; x++) {
                const offset = (Math.floor(y / scale) * 16 + Math.floor(x / scale)) * 3
                if (Math.abs(image.red(x, y) - expected[offset]) > 2) {
                    image.save(TestConfig.fixtureRoot + "/mismatch.png")
                    fail("Red pixel mismatch at " + x + "," + y + ": " + image.red(x, y)
                        + " expected " + expected[offset] + " scale " + scale)
                }
                verify(Math.abs(image.green(x, y) - expected[offset + 1]) <= 2, "Green pixel mismatch")
                verify(Math.abs(image.blue(x, y) - expected[offset + 2]) <= 2, "Blue pixel mismatch")
                compare(image.alpha(x, y), 255)
            }
        }
    }

    function test_cached_pixels_survive_neighbor_redraws_and_prefetch() {
        const item = shader()
        const rendered = createTemporaryObject(spyComponent, tests, {target: findChild(item, "composedFrame")})
        rendered.clear()
        checkPixels(item, "stills/0000", 1)
        createTemporaryObject(neighborComponent, tests)
        wait(400)
        checkPixels(item, "stills/0000", 1)
        compare(rendered.count, 0)

        item.nextIndex = 1
        item.transitionFrame = 1
        tryVerify(function() { return item.displayedKey.indexOf("transitions/0000-0001/0001") >= 0 }, 2000)
        tryCompare(item, "loading", false, 2000)
        compare(rendered.count, 1)
        checkPixels(item, "transitions/0000-0001/0001", 1)
        wait(200)
        compare(rendered.count, 1)
    }

    function test_saved_state_renders_correct_pixels_without_a_service() {
        const renderer = createTemporaryObject(clientComponent, tests, {
            requestSettings: {images: ["/pending.png"], frames: 8},
            playbackSettings: {mode: "Image"}, cachedDirectory: TestConfig.fixtureRoot,
            cachedState: JSON.stringify({directory: TestConfig.fixtureRoot, generation: "offline",
                manifest: TestConfig.manifest, settings: {images: ["/one.png", "/two.png"], frames: 4},
                image_count: 2, frame_count: 4, playback: {mode: "Slideshow"}})
        })
        const item = createTemporaryObject(shaderComponent, tests, {
            renderDirectory: renderer.displayDirectory, generation: renderer.displayGeneration,
            manifest: renderer.displayManifest, imageCount: renderer.displayImageCount,
            transitionFrames: renderer.displayFrameCount
        })
        tryCompare(item, "ready", true, 2000)
        tryCompare(item, "loading", false, 2000)
        compare(renderer.renderState, "ready")
        compare(renderer.jobId, "")
        verify(!renderer.requestInFlight)
        compare(item.displayGeneration, "offline")
        compare(item.displayImageCount, 2)
        compare(item.displayFrameCount, 4)
        checkPixels(item, "stills/0000", 1)
    }

    function test_resize_and_generation_refresh_the_output() {
        const item = shader()
        const rendered = createTemporaryObject(spyComponent, tests, {target: findChild(item, "composedFrame")})
        rendered.clear()
        item.width = 32
        item.height = 16
        tryCompare(rendered, "count", 1, 2000)
        tryCompare(item, "loading", false, 2000)
        checkPixels(item, "stills/0000", 2)
        item.generation = "replacement"
        tryCompare(item, "displayGeneration", "replacement", 2000)
        compare(rendered.count, 2)
        checkPixels(item, "stills/0000", 2)
    }

    function test_incomplete_replacement_keeps_the_cached_pixels() {
        const item = shader()
        item.renderDirectory = TestConfig.fixtureRoot + "/missing"
        item.generation = "missing"
        wait(100)
        verify(item.ready)
        verify(item.loading)
        compare(item.displayGeneration, "initial")
        checkPixels(item, "stills/0000", 1)
        item.renderDirectory = ""
        verify(!item.ready)
    }

    function test_thumbnail_downsampling_preserves_glyph_coverage_and_colors() {
        if (Screen.devicePixelRatio !== 1) skip("Pixel averages use a 1x thumbnail")
        const item = createTemporaryObject(shaderComponent, tests, {
            width: 4, height: 2, outputTextureSize: Qt.size(16, 8),
            smoothOutput: true, mipmapOutput: true,
            renderDirectory: TestConfig.fixtureRoot, generation: "thumbnail",
            manifest: TestConfig.manifest, imageCount: 2
        })
        tryCompare(item, "ready", true, 2000)
        tryCompare(item, "loading", false, 2000)
        const image = grabImage(item)
        const expected = TestConfig.pixels["stills/0000"]
        compare(image.width, 4)
        compare(image.height, 2)
        for (let y = 0; y < 2; y++) {
            for (let x = 0; x < 4; x++) {
                const average = [0, 0, 0]
                for (let dy = 0; dy < 4; dy++) {
                    for (let dx = 0; dx < 4; dx++) {
                        const offset = ((y * 4 + dy) * 16 + x * 4 + dx) * 3
                        for (let channel = 0; channel < 3; channel++)
                            average[channel] += expected[offset + channel] / 16
                    }
                }
                verify(Math.abs(image.red(x, y) - average[0]) <= 2, "Red thumbnail coverage")
                verify(Math.abs(image.green(x, y) - average[1]) <= 2, "Green thumbnail coverage")
                verify(Math.abs(image.blue(x, y) - average[2]) <= 2, "Blue thumbnail coverage")
            }
        }
    }
}
