/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick
import QtTest
import TestSupport 1.0
import Wallpaper 1.0

TestCase {
    id: tests
    name: "VideoPreview"
    when: windowShown
    visible: true
    width: 900
    height: 650

    Component { id: configComponent; Config { width: 850 } }

    function test_video_preview_checks_source_and_keeps_excerpt_out_of_saved_wallpaper() {
        const item = createTemporaryObject(configComponent, tests,
            {cfg_Mode: "Video", cfg_Images: ["/wrong-kind.jpg"]})
        verify(item)
        findChild(item, "renderClient").serviceUrl = TestConfig.serviceUrl
        findChild(item, "cacheClient").serviceUrl = TestConfig.serviceUrl
        tryCompare(item, "controlsReady", true)
        const preview = findChild(item, "previewChangesButton")
        const count = findChild(item, "videoWorkEstimate")
        const source = findChild(item, "sourceCheck")
        tryCompare(source, "status", "error")
        verify(!preview.enabled)
        item.setProfileSetting("Images", ["/preview-video.mkv"])
        item.setProfileSetting("FrameStride", 3)
        item.setProfileSetting("VideoTransitionFrames", 1)
        tryCompare(source, "status", "valid")

        verify(preview.enabled)
        verify(count.text.includes("40 source frames") && count.text.includes("79 frames"))
        preview.clicked()
        const client = findChild(item, "renderClient")
        tryCompare(client, "displayGeneration", "excerpt", 3000)
        compare(item.cfg_RenderDirectory, "")
        item.saveConfig()
        tryCompare(client, "displayGeneration", "full", 3000)
        compare(JSON.parse(item.cfg_RenderedState).generation, "full")
    }
}
