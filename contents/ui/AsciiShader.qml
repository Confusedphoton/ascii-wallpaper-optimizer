/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick

Item {
    id: asciiRoot

    property string renderDirectory: ""
    property string generation: ""
    property var manifest: ({})
    property int currentIndex: 0
    property int nextIndex: 0
    property real transitionProgress: 0.0
    property int transitionFrames: 16
    property int imageCount: 1
    property int activeBuffer: -1
    readonly property var displayedFrame: activeBuffer === 0 ? firstFrame : activeBuffer === 1 ? secondFrame : null
    readonly property bool ready: !!displayedFrame && displayedFrame.ready
    readonly property string displayGeneration: ready ? displayedFrame.descriptor.generation : ""
    readonly property string displayedKey: ready ? displayedFrame.frameKey : ""
    readonly property int displayImageCount: ready ? Number(displayedFrame.manifest.image_count)
        || displayedFrame.descriptor.imageCount : 0
    readonly property int displayFrameCount: ready ? Number(displayedFrame.manifest.frame_count)
        || displayedFrame.descriptor.frameCount : 0
    readonly property bool loading: !!renderDirectory && displayedKey !== desiredKey
    readonly property int frameCount: Math.max(2, Number(manifest.frame_count) || transitionFrames)
    readonly property int frameIndex: Math.max(0, Math.min(frameCount - 1,
        Math.floor(transitionProgress * (frameCount - 1))))
    readonly property string desiredKey: descriptor(currentIndex, nextIndex, frameIndex).key
    signal framePresented()

    onDesiredKeyChanged: refresh.restart()
    onManifestChanged: refresh.restart()
    onRenderDirectoryChanged: {
        if (!renderDirectory) activeBuffer = -1
    }
    Component.onCompleted: refresh.restart()

    function descriptor(current, next, index) {
        const directory = String(renderDirectory || "").trim().replace(/\/$/, "")
        let path = "stills/" + String(current).padStart(4, "0")
        if (current !== next && index > 0 && index < frameCount - 1) {
            const pair = String(current).padStart(4, "0") + "-" + String(next).padStart(4, "0")
            path = "transitions/" + pair + "/" + String(index).padStart(4, "0")
        } else if (current !== next && index === frameCount - 1) {
            path = "stills/" + String(next).padStart(4, "0")
        }
        const version = generation || String(manifest.generation || directory)
        return {key: directory + "/" + path + "?v=" + version,
            url: "file://" + directory.split("/").map(encodeURIComponent).join("/"),
            generation: version, path: path, manifest: manifest,
            imageCount: Math.max(1, Number(manifest.image_count) || imageCount), frameCount: frameCount}
    }

    function updateFrame() {
        if (!renderDirectory) return
        if (displayedKey === desiredKey) { prefetch(); return }
        const buffer = activeBuffer === 0 ? secondFrame : firstFrame
        buffer.load(descriptor(currentIndex, nextIndex, frameIndex))
        present(buffer)
    }

    function present(buffer) {
        if (!renderDirectory || !buffer.ready || buffer.frameKey !== desiredKey) return
        const nextBuffer = buffer === firstFrame ? 0 : 1
        if (activeBuffer !== nextBuffer) {
            activeBuffer = nextBuffer
            framePresented()
        }
        Qt.callLater(prefetch)
    }

    function prefetch() {
        if (!renderDirectory || !ready || displayedKey !== desiredKey) return
        let upcoming
        if (currentIndex !== nextIndex && frameIndex < frameCount - 1)
            upcoming = descriptor(currentIndex, nextIndex, frameIndex + 1)
        else if (currentIndex === nextIndex && displayImageCount > 1)
            upcoming = descriptor(currentIndex, (currentIndex + 1) % displayImageCount, 1)
        if (upcoming && upcoming.key !== desiredKey) {
            const buffer = activeBuffer === 0 ? secondFrame : firstFrame
            buffer.load(upcoming)
        }
    }

    Timer { id: refresh; interval: 0; onTriggered: asciiRoot.updateFrame() }
    AsciiFrame {
        id: firstFrame
        anchors.fill: parent
        active: asciiRoot.activeBuffer === 0
        onCompleted: asciiRoot.present(firstFrame)
    }
    AsciiFrame {
        id: secondFrame
        anchors.fill: parent
        active: asciiRoot.activeBuffer === 1
        onCompleted: asciiRoot.present(secondFrame)
    }
}
