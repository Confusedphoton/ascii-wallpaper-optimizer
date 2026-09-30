/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import "../../contents/ui" as Wallpaper

TestCase {
    id: tests
    name: "RenderClient"
    when: windowShown

    Component {
        id: clientComponent
        Wallpaper.RenderClient {
            autoStart: false
            serviceUrl: TestConfig.serviceUrl
            requestDelay: 20
            requestTimeout: 1000
            retryBaseInterval: 40
            retryMaxInterval: 80
            readyStatusInterval: 10000
            statusInterval: 10000
        }
    }
    SignalSpy { id: completed; signalName: "assetsReady" }

    function client(image) {
        return createTemporaryObject(clientComponent, tests,
            {requestSettings: {images: [image], frames: 4}, playbackSettings: {mode: "Image"}})
    }

    function test_cached_assets_survive_pending_settings() {
        const renderer = createTemporaryObject(clientComponent, tests,
            {requestSettings: {images: ["/one.png", "/two.png"], frames: 4},
                playbackSettings: {mode: "Slideshow", slideInterval: 10}, cachedDirectory: "/cached"})
        compare(renderer.displayDirectory, "/cached")
        compare(renderer.displayImageCount, 2)
        compare(renderer.displayFrameCount, 4)
        renderer.requestSettings = {images: ["/new.png"], frames: 8}
        renderer.playbackSettings = {mode: "Image", slideInterval: 5}
        compare(renderer.displayDirectory, "/cached")
        compare(renderer.displayImageCount, 2)
        compare(renderer.displayFrameCount, 4)
        compare(renderer.displayPlaybackSettings.mode, "Slideshow")
    }

    function test_pending_settings_keep_published_assets() {
        const renderer = client("/one.png")
        completed.target = renderer
        completed.clear()
        renderer.consume({id: "job", state: "ready", directory: "/old", generation: "first",
            image_count: 2, frame_count: 4, settings: {images: ["/one.png", "/two.png"], frames: 4},
            manifest: {image_count: 2}})
        compare(completed.count, 1)
        renderer.consume({id: "job", state: "queued", directory: "/new", generation: "next"})
        compare(renderer.displayDirectory, "/old")
        compare(renderer.displayFrameCount, 4)
        compare(renderer.displayImageCount, 2)
        compare(renderer.displayManifest.image_count, 2)
        renderer.requestSettings = {images: ["/new.png"], frames: 8}
        compare(renderer.displayDirectory, "/old")
        compare(renderer.displayFrameCount, 4)
        renderer.consume({id: "newjob", state: "ready", directory: "/new", generation: "next",
            image_count: 1, frame_count: 8, settings: {images: ["/new.png"], frames: 8},
            manifest: {image_count: 1}})
        compare(completed.count, 2)
        compare(renderer.displayDirectory, "/new")
        compare(renderer.displayFrameCount, 8)
        renderer.consume({id: "newjob", state: "ready", directory: "/new", generation: "next"})
        compare(completed.count, 2)
        compare(renderer.displayManifest.image_count, 1)
    }

    function test_connection_retries_automatically() {
        const renderer = client("/retry.png")
        renderer.start(false)
        tryCompare(renderer, "renderState", "ready", 3000)
        compare(renderer.displayGeneration, "retry-ready")
        compare(renderer.retryAttempt, 0)
        verify(!renderer.requestInFlight)
        compare(renderer.error, "")
    }

    function test_queued_render_is_polled_until_its_assets_are_published() {
        const renderer = client("/queued.png")
        renderer.statusInterval = 20
        renderer.start(false)
        tryCompare(renderer, "renderState", "ready", 3000)
        compare(renderer.displayGeneration, "queued-ready")
        verify(renderer.displayDirectory.length > 0)
    }

    function test_request_timeout_ignores_late_reply() {
        const renderer = client("/timeout.png")
        renderer.requestTimeout = 100
        renderer.start(false)
        tryCompare(renderer, "displayGeneration", "timeout-ready", 3000)
        wait(350)
        compare(renderer.displayGeneration, "timeout-ready")
        verify(!renderer.requestInFlight)
    }

    function test_status_has_inflight_guard() {
        const renderer = client("/slow-status.png")
        renderer.start(false)
        tryCompare(renderer, "renderState", "ready", 2000)
        renderer.checkStatus()
        verify(renderer.requestInFlight)
        const serial = renderer.requestSerial
        renderer.checkStatus()
        compare(renderer.requestSerial, serial)
        tryCompare(renderer, "requestInFlight", false, 2000)
        compare(renderer.displayGeneration, "slow-status-ready")
    }

    function test_ready_poll_observes_forced_generation() {
        const renderer = client("/refresh.png")
        renderer.readyStatusInterval = 40
        completed.target = renderer
        completed.clear()
        renderer.start(false)
        tryCompare(renderer, "displayGeneration", "refresh-new", 3000)
        compare(completed.count, 2)
        verify(renderer.displayDirectory.endsWith("/new"))
        compare(renderer.displayManifest.image_count, 1)
    }
}
