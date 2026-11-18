/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import "../../contents/ui" as Wallpaper

TestCase {
    id: tests
    name: "FlowPlayback"
    when: windowShown
    visible: true
    width: 32; height: 32
    Component { id: loopComponent; Wallpaper.FlowLoopPlayer {} }
    Component { id: controllerComponent; Wallpaper.PlaybackController {} }
    Component { id: shaderComponent; Wallpaper.AsciiShader { width: 16; height: 16 } }

    function manifest(count) {
        return {schema_version: 2, image_count: count, frame_count: 4,
            output_size: [4, 4], cell_size: [2, 2], atlas_count: 1,
            loops: {"0000": {path: "loops/0000", frame_count: 4, duration_ms: 200},
                    "0001": {path: "loops/0001", frame_count: 4, duration_ms: 200}}}
    }

    function test_single_image_wrap_and_loading_pause() {
        const player = createTemporaryObject(loopComponent, tests,
            {playing: true, ready: true, frameCount: 4, duration: 400})
        const frames = []
        player.frameIndexChanged.connect(function() { frames.push(player.frameIndex) })
        tryCompare(player, "frameIndex", 1, 500)
        player.loading = true
        verify(!player.timerRunning)
        wait(200)
        compare(player.frameIndex, 1)
        player.loading = false
        tryVerify(function() { return player.ticks >= 8 }, 1500)
        compare(frames.slice(0, 8), [1, 2, 3, 0, 1, 2, 3, 0])
        player.clipKey = "replacement"
        compare(player.frameIndex, 0)
        compare(player.ticks, 0)
        player.playing = false
        wait(150)
        compare(player.frameIndex, 0)
    }

    function test_slideshow_starts_at_presented_loop_boundary() {
        const controller = createTemporaryObject(controllerComponent, tests,
            {ready: true, imageCount: 2, manifest: manifest(2), slideshowEnabled: true,
             slideInterval: 250, transitionDuration: 200, frameCount: 4})
        tryVerify(function() { return controller.loopFrame > 0 }, 500)
        compare(controller.currentIndex, 0)
        compare(controller.nextIndex, 0)
        const phases = []
        controller.nextIndexChanged.connect(function() { phases.push(controller.loopFrame) })
        wait(220)
        compare(controller.nextIndex, 0)
        tryCompare(controller, "nextIndex", 1, 600)
        compare(phases, [-1])
        tryCompare(controller, "currentIndex", 1, 1000)
        controller.slideshowEnabled = false
        controller.reset()
        compare(controller.currentIndex, 0)
    }

    function test_late_timer_shifts_clock_without_bursting_frames() {
        const player = createTemporaryObject(loopComponent, tests,
            {playing: true, ready: true, frameCount: 4, duration: 400})
        tryVerify(function() { return player.startedAt > 0 }, 500)
        const until = Date.now() + 350
        while (Date.now() < until) { /* Simulate a busy compositor/UI thread. */ }
        tryCompare(player, "ticks", 1, 300)
        verify(player.pausedDuration > 200)
        wait(40)
        compare(player.ticks, 1)
        tryCompare(player, "ticks", 2, 300)
    }

    function test_loop_descriptor_wrap_and_missing_frame_retains_display() {
        const shader = createTemporaryObject(shaderComponent, tests,
            {renderDirectory: TestConfig.fixtureRoot + "/old", generation: "loop",
             manifest: manifest(1), loopFrame: 3})
        tryCompare(shader, "loading", false, 2000)
        verify(shader.displayedKey.indexOf("loops/0000/0003") >= 0)
        shader.loopFrame = 0
        tryVerify(function() { return shader.displayedKey.indexOf("loops/0000/0000") >= 0 }, 2000)
        const displayed = shader.displayedKey
        shader.renderDirectory = TestConfig.fixtureRoot + "/partial"
        shader.generation = "partial-loop"
        tryVerify(function() { return !!shader.frameError }, 2000)
        verify(shader.ready)
        compare(shader.displayedKey, displayed)
        compare(shader.displayGeneration, "loop")
    }

    function videoManifest() {
        return {schema_version: 3, image_count: 1, frame_count: 4,
            output_size: [4, 4], cell_size: [2, 2], atlas_count: 1,
            video: {path: "video", frame_count: 4, duration_ms: 650,
                    frame_durations_ms: [200, 50, 200, 200]}}
    }

    function test_video_skips_expired_holds_when_playback_falls_behind() {
        const clip = videoManifest()
        clip.video.duration_ms = 1000
        clip.video.frame_durations_ms = [300, 50, 350, 300]
        const controller = createTemporaryObject(controllerComponent, tests,
            {ready: true, manifest: clip, generation: "video"})
        const frames = []
        controller.loopFrameChanged.connect(function() { frames.push(controller.loopFrame) })
        wait(50)
        compare(controller.loopFrame, 0)
        controller.loading = true
        const until = Date.now() + 350
        while (Date.now() < until) { /* Simulate a busy compositor/UI thread. */ }
        tryCompare(controller, "loopFrame", 2, 250)
        compare(frames[0], 2)
        controller.loading = false
        tryCompare(controller, "loopFrame", 3, 600)
        tryCompare(controller, "loopFrame", 0, 600)
        controller.generation = "replacement-video"
        compare(controller.loopFrame, 0)
    }

    function test_video_descriptor_wrap_and_missing_frame_retains_display() {
        const shader = createTemporaryObject(shaderComponent, tests,
            {renderDirectory: TestConfig.fixtureRoot + "/old", generation: "video",
             manifest: videoManifest(), loopFrame: 3})
        tryCompare(shader, "loading", false, 2000)
        verify(shader.displayedKey.indexOf("video/0003") >= 0)
        shader.loopFrame = 0
        tryVerify(function() { return shader.displayedKey.indexOf("video/0000") >= 0 }, 2000)
        const displayed = shader.displayedKey
        shader.renderDirectory = TestConfig.fixtureRoot + "/partial"
        shader.generation = "partial-video"
        tryVerify(function() { return !!shader.frameError }, 2000)
        compare(shader.displayedKey, displayed)
    }

    function test_single_frame_video_holds_without_a_timer() {
        const clip = videoManifest()
        clip.video.frame_count = 1
        clip.video.frame_durations_ms = [650]
        const controller = createTemporaryObject(controllerComponent, tests,
            {ready: true, manifest: clip})
        verify(controller.hasLoop)
        compare(controller.loopFrame, 0)
        wait(100)
        compare(controller.loopFrame, 0)
    }

    function test_switch_between_still_flow_and_video_manifests() {
        const controller = createTemporaryObject(controllerComponent, tests,
            {ready: true, enabled: false})
        compare(controller.hasLoop, false)
        controller.manifest = videoManifest()
        compare(controller.hasLoop, true)
        compare(controller.loopInfo.frame_count, 4)
        controller.manifest = manifest(1)
        compare(controller.hasLoop, true)
        compare(controller.loopInfo.path, "loops/0000")
        controller.manifest = ({})
        compare(controller.hasLoop, false)
        compare(controller.loopFrame, -1)
    }
}
