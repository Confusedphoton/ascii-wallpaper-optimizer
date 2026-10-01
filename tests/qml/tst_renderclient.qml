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
            serviceUrl: TestConfig.serviceUrl
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

    function test_startup_and_settings_changes_never_submit_a_render() {
        const renderer = client("/manual-only.png")
        const serial = renderer.requestSerial
        wait(100)
        compare(renderer.requestSerial, serial)
        compare(renderer.renderState, "idle")
        verify(!renderer.requestInFlight)
        renderer.requestSettings = {images: ["/manual-only-updated.png"], frames: 8}
        const changedSerial = renderer.requestSerial
        wait(100)
        compare(renderer.requestSerial, changedSerial)
        compare(renderer.jobId, "")
        verify(!renderer.submissionPending)
        renderer.start(true)
        tryCompare(renderer, "renderState", "ready", 3000)
        compare(renderer.displayGeneration, "manual-only-updated-ready")
    }

    function test_superseded_job_requires_an_explicit_retry() {
        const renderer = client("/superseded.png")
        renderer.consume({id: "superseded", state: "superseded"})
        const serial = renderer.requestSerial
        wait(100)
        compare(renderer.requestSerial, serial)
        compare(renderer.jobId, "")
        compare(renderer.renderState, "idle")
    }

    function test_saved_cache_reconnects_with_read_only_status() {
        const key = "0123456789abcdef01234567"
        const directory = TestConfig.fixtureRoot + "/variants/" + key + "/0123456789abcdef0123456789abcdef"
        const renderer = createTemporaryObject(clientComponent, tests, {
            requestSettings: {images: ["/saved-monitor.png"], frames: 4},
            cachedDirectory: directory,
            cachedState: JSON.stringify({directory: directory, generation: "saved",
                settings: {images: ["/saved-monitor.png"], frames: 4}}),
            readyStatusInterval: 40
        })
        compare(renderer.displayDirectory, directory)
        compare(renderer.jobId, key)
        tryCompare(renderer, "displayGeneration", key + "-ready", 3000)
        verify(!renderer.submissionPending)
    }

    function test_switching_saved_profiles_updates_the_observed_job() {
        const first = TestConfig.fixtureRoot + "/variants/0123456789abcdef01234567/0123456789abcdef0123456789abcdef"
        const second = TestConfig.fixtureRoot + "/variants/fedcba9876543210fedcba98/fedcba9876543210fedcba9876543210"
        const renderer = createTemporaryObject(clientComponent, tests, {
            requestSettings: {images: ["/profiles.png"], frames: 4}, cachedDirectory: first,
            cachedState: JSON.stringify({directory: first, generation: "first",
                settings: {images: ["/profiles.png"], frames: 4}})
        })
        compare(renderer.jobId, "0123456789abcdef01234567")
        renderer.cachedState = JSON.stringify({directory: second, generation: "second",
            settings: {images: ["/profiles.png"], frames: 4}})
        renderer.cachedDirectory = second
        tryCompare(renderer, "displayDirectory", second, 1000)
        compare(renderer.jobId, "fedcba9876543210fedcba98")
        compare(renderer.renderState, "ready")
        verify(!renderer.submissionPending)
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

    function test_saved_generation_restores_without_the_service() {
        const directory = TestConfig.fixtureRoot + "/old"
        const saved = {directory: directory, generation: "saved-generation",
            settings: {images: ["/one.png", "/two.png"], frames: 4},
            image_count: 2, frame_count: 4,
            manifest: {image_count: 2, frame_count: 4, invert_ink: true},
            playback: {mode: "Slideshow", slideInterval: 10}}
        const renderer = createTemporaryObject(clientComponent, tests,
            {requestSettings: {images: ["/pending.png"], frames: 8},
                playbackSettings: {mode: "Image"}, cachedDirectory: directory,
                cachedState: JSON.stringify(saved), serviceUrl: "http://127.0.0.1:1"})
        compare(renderer.renderState, "ready")
        compare(renderer.displayGeneration, "saved-generation")
        compare(renderer.displayImageCount, 2)
        compare(renderer.displayFrameCount, 4)
        compare(renderer.displayPlaybackSettings.mode, "Slideshow")
        verify(renderer.displayManifest.invert_ink)
        const serial = renderer.requestSerial
        wait(50)
        compare(renderer.requestSerial, serial)
        verify(!renderer.requestInFlight)
    }

    function test_corrupt_saved_state_uses_legacy_cache() {
        const renderer = createTemporaryObject(clientComponent, tests,
            {requestSettings: {images: ["/one.png"], frames: 4},
                cachedDirectory: "/cached", cachedState: "invalid JSON"})
        compare(renderer.renderState, "ready")
        compare(renderer.displayDirectory, "/cached")
        compare(renderer.displayImageCount, 1)
    }

    function test_playback_changes_are_included_in_saved_state() {
        const renderer = client("/one.png")
        completed.target = renderer
        completed.clear()
        renderer.consume({id: "job", state: "ready", directory: "/cached", generation: "saved",
            settings: {images: ["/one.png"], frames: 4}, image_count: 1, frame_count: 4})
        renderer.playbackSettings = {mode: "Slideshow", slideInterval: 25}
        tryCompare(completed, "count", 2, 1000)
        compare(JSON.parse(renderer.publishedState).playback.slideInterval, 25)
        renderer.renderState = "connecting"
        renderer.playbackSettings = {mode: "Slideshow", slideInterval: 30}
        renderer.consume({id: "job", state: "ready", directory: "/cached", generation: "saved"})
        compare(completed.count, 3)
        compare(JSON.parse(renderer.publishedState).playback.slideInterval, 30)
        renderer.consume({id: "job", state: "ready", directory: "/cached", generation: "saved"})
        compare(completed.count, 3)
    }

    function test_missing_status_retains_saved_assets_without_resubmitting() {
        const renderer = client("/restart.png")
        renderer.retryBaseInterval = 200
        renderer.retryMaxInterval = 500
        renderer.start(true)
        tryCompare(renderer, "renderState", "ready", 3000)
        verify(!renderer.currentRequestForced)
        renderer.checkStatus()
        tryCompare(renderer, "jobId", "", 3000)
        verify(renderer.displayDirectory.length > 0)
        tryCompare(renderer, "renderState", "ready", 3000)
        compare(renderer.error, "")
        verify(!renderer.requestInFlight)
        const serial = renderer.requestSerial
        wait(600)
        compare(renderer.requestSerial, serial)
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
