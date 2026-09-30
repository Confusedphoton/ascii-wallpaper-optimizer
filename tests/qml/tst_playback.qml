/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import "../../contents/ui" as Wallpaper

TestCase {
    id: tests
    name: "Playback"
    when: windowShown
    visible: true
    width: 32
    height: 32

    Component { id: playerComponent; Wallpaper.SlideshowPlayer {} }
    Component { id: shaderComponent; Wallpaper.AsciiShader { width: 16; height: 16 } }

    function manifest(imageCount) {
        return {image_count: imageCount, frame_count: 4, output_size: [4, 4],
            cell_size: [2, 2], atlas_count: 1, invert_ink: false,
            background_labs: {"stills/0000": [0.2, 0, 0]}}
    }

    function test_timer_gates_remain_bound_after_reset() {
        const player = createTemporaryObject(playerComponent, tests,
            {ready: false, imageCount: 2, slideInterval: 10000})
        verify(!player.timerRunning)
        player.ready = true
        verify(player.timerRunning)
        player.reset()
        verify(player.timerRunning)
        player.enabled = false
        player.reset()
        wait(20)
        verify(!player.timerRunning)
        player.enabled = true
        verify(player.timerRunning)
        player.imageCount = 1
        verify(!player.timerRunning)
        player.reset()
        verify(!player.timerRunning)
    }

    function test_animation_waits_for_complete_frames() {
        const player = createTemporaryObject(playerComponent, tests,
            {ready: true, imageCount: 2, slideInterval: 10000, transitionDuration: 180})
        player.advance()
        compare(player.nextIndex, 1)
        verify(player.animating)
        wait(40)
        player.loading = true
        verify(player.paused)
        const held = player.transitionProgress
        wait(70)
        compare(player.transitionProgress, held)
        player.loading = false
        tryCompare(player, "currentIndex", 1, 1000)
        compare(player.transitionProgress, 0)
        verify(player.timerRunning)
        player.advance()
        player.generation = "new"
        compare(player.currentIndex, 0)
        compare(player.nextIndex, 0)
        compare(player.transitionProgress, 0)
        verify(!player.animating)
    }

    function test_transition_presents_every_discrete_frame() {
        const player = createTemporaryObject(playerComponent, tests,
            {ready: true, imageCount: 2, frameCount: 4, slideInterval: 10000, transitionDuration: 240})
        const frames = []
        player.transitionFrameChanged.connect(function() { frames.push(player.transitionFrame) })
        player.advance()
        wait(40)
        compare(player.transitionFrame, 0)
        compare(player.transitionProgress, 0)
        tryCompare(player, "currentIndex", 1, 1000)
        compare(frames, [1, 2, 3, 0])
        verify(!player.animating)
    }

    function test_ten_second_frame_hold_does_not_update_between_boundaries() {
        const player = createTemporaryObject(playerComponent, tests,
            {ready: true, imageCount: 2, frameCount: 4, slideInterval: 60000, transitionDuration: 30000})
        const frames = []
        player.transitionFrameChanged.connect(function() { frames.push(player.transitionFrame) })
        player.advance()
        wait(1000)
        compare(player.transitionFrame, 0)
        compare(frames, [])
        tryCompare(player, "transitionFrame", 1, 11000)
        compare(frames, [1])
        wait(100)
        compare(player.transitionFrame, 1)
        compare(frames, [1])
        player.reset()
    }

    function test_loading_preserves_the_remaining_frame_hold() {
        const player = createTemporaryObject(playerComponent, tests,
            {ready: true, imageCount: 2, frameCount: 3, slideInterval: 10000, transitionDuration: 2000})
        player.advance()
        wait(600)
        player.loading = true
        wait(500)
        compare(player.transitionFrame, 0)
        player.loading = false
        tryCompare(player, "transitionFrame", 1, 700)
        player.reset()
    }

    function test_legacy_metadata_fallback_loads_cached_assets() {
        const shader = createTemporaryObject(shaderComponent, tests, {
            renderDirectory: TestConfig.fixtureRoot + "/old", imageCount: 2, transitionFrames: 4
        })
        tryCompare(shader, "ready", true, 2000)
        compare(shader.displayImageCount, 2)
        compare(shader.displayFrameCount, 4)
        verify(!shader.loading)
    }

    function test_atomic_frames_and_generation_refresh() {
        const shader = createTemporaryObject(shaderComponent, tests, {
            renderDirectory: TestConfig.fixtureRoot + "/old", generation: "old",
            manifest: manifest(2), imageCount: 2
        })
        tryCompare(shader, "ready", true, 2000)
        compare(shader.displayGeneration, "old")
        const initial = shader.displayedKey
        shader.nextIndex = 1
        shader.transitionProgress = 0.5
        tryVerify(function() { return shader.displayedKey !== initial }, 2000)
        verify(shader.ready)
        verify(shader.displayedKey.indexOf("transitions/0000-0001/0001") >= 0)
        const displayed = shader.displayedKey
        shader.currentIndex = 0
        shader.nextIndex = 0
        shader.renderDirectory = TestConfig.fixtureRoot + "/partial"
        shader.generation = "partial"
        wait(150)
        verify(shader.ready)
        verify(shader.loading)
        compare(shader.displayedKey, displayed)
        compare(shader.displayGeneration, "old")
        shader.renderDirectory = TestConfig.fixtureRoot + "/new"
        shader.generation = "new"
        shader.manifest = manifest(1)
        tryCompare(shader, "displayGeneration", "new", 2000)
        verify(shader.ready)
        verify(!shader.loading)
        compare(shader.displayImageCount, 1)
        shader.generation = "forced"
        tryCompare(shader, "displayGeneration", "forced", 2000)
        verify(shader.ready)
    }
}
