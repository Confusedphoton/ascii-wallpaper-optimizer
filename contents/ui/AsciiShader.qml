/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick
import QtQuick.Window

Item {
    id: asciiRoot

    property string renderDirectory: ""
    property string generation: ""
    property var manifest: ({})
    property int currentIndex: 0
    property int nextIndex: 0
    property real transitionProgress: 0.0
    property int transitionFrame: -1
    property int transitionFrames: 16
    property int imageCount: 1
    property int loopFrame: -1
    property string frameError: ""
    property int activeBuffer: -1
    property var cachedDescriptor: null
    property var pendingDescriptor: null
    property bool outputDirty: false
    readonly property var displayedFrame: activeBuffer === 0 ? firstFrame : activeBuffer === 1 ? secondFrame : null
    readonly property bool ready: !!cachedDescriptor
    readonly property string displayGeneration: ready ? cachedDescriptor.generation : ""
    readonly property string displayedKey: ready ? cachedDescriptor.key : ""
    readonly property int displayImageCount: ready ? cachedDescriptor.imageCount : 0
    readonly property int displayFrameCount: ready ? cachedDescriptor.frameCount : 0
    readonly property bool loading: !!renderDirectory && (displayedKey !== desiredKey || !!pendingDescriptor)
    readonly property int frameCount: Math.max(2, Number(manifest.frame_count) || transitionFrames)
    readonly property int frameIndex: Math.max(0, Math.min(frameCount - 1,
        transitionFrame >= 0 ? transitionFrame : Math.floor(transitionProgress * (frameCount - 1))))
    property bool smoothOutput: false
    property bool mipmapOutput: false
    property size outputTextureSize: Qt.size(Math.max(1, Math.round(width * Screen.devicePixelRatio)),
        Math.max(1, Math.round(height * Screen.devicePixelRatio)))
    readonly property string desiredKey: descriptor(currentIndex, nextIndex, frameIndex).key
    signal framePresented()

    onDesiredKeyChanged: { frameError = ""; refresh.restart() }
    onManifestChanged: refresh.restart()
    onOutputTextureSizeChanged: {
        outputDirty = true
        refresh.restart()
    }
    onRenderDirectoryChanged: {
        if (!renderDirectory) {
            activeBuffer = -1
            cachedDescriptor = null
            pendingDescriptor = null
        }
    }
    Component.onCompleted: refresh.restart()

    function descriptor(current, next, index) {
        const directory = String(renderDirectory || "").trim().replace(/\/$/, "")
        let path = "stills/" + String(current).padStart(4, "0")
        const loop = manifest.schema_version === 2
            ? (manifest.loops || ({}))[String(current).padStart(4, "0")] : null
        if (current === next && loopFrame >= 0 && loop
                && loop.path === "loops/" + String(current).padStart(4, "0") && loop.frame_count >= 4)
            path = loop.path + "/" + String(Math.min(loop.frame_count - 1, loopFrame)).padStart(4, "0")
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
        if (!renderDirectory || pendingDescriptor || width < 1 || height < 1) return
        if (displayedKey === desiredKey) {
            if (outputDirty && displayedFrame && displayedFrame.ready) capture(displayedFrame.descriptor)
            else prefetch()
            return
        }
        if (displayedFrame && displayedFrame.ready && displayedFrame.frameKey === desiredKey) {
            present(displayedFrame)
            return
        }
        const buffer = activeBuffer === 0 ? secondFrame : firstFrame
        buffer.load(descriptor(currentIndex, nextIndex, frameIndex))
        present(buffer)
    }

    function present(buffer) {
        if (!renderDirectory || pendingDescriptor || !buffer.ready || buffer.frameKey !== desiredKey
                || width < 1 || height < 1 || displayedKey === desiredKey) return
        const nextBuffer = buffer === firstFrame ? 0 : 1
        activeBuffer = nextBuffer
        capture(buffer.descriptor)
    }

    function capture(value) {
        // Freeze the source buffer until this render finishes. The existing
        // output texture remains visible while a replacement is loading.
        pendingDescriptor = value
        outputDirty = false
        composedFrame.scheduleUpdate()
    }

    function finishCapture() {
        if (!pendingDescriptor || !renderDirectory) return
        const changed = displayedKey !== pendingDescriptor.key
        cachedDescriptor = pendingDescriptor
        pendingDescriptor = null
        if (changed) framePresented()
        refresh.restart()
    }

    function prefetch() {
        if (!renderDirectory || pendingDescriptor || !ready || displayedKey !== desiredKey) return
        let upcoming
        if (currentIndex === nextIndex && loopFrame >= 0) {
            const loop = (manifest.loops || ({}))[String(currentIndex).padStart(4, "0")]
            if (loop) {
                upcoming = descriptor(currentIndex, currentIndex, 0)
                upcoming.path = loop.path + "/" + String((loopFrame + 1) % loop.frame_count).padStart(4, "0")
                upcoming.key = String(renderDirectory).replace(/\/$/, "") + "/" + upcoming.path + "?v=" + upcoming.generation
            }
        } else if (currentIndex !== nextIndex && frameIndex < frameCount - 1)
            upcoming = descriptor(currentIndex, nextIndex, frameIndex + 1)
        else if (currentIndex === nextIndex && displayImageCount > 1)
            upcoming = descriptor(currentIndex, (currentIndex + 1) % displayImageCount, 1)
        if (upcoming && upcoming.key !== desiredKey) {
            const buffer = activeBuffer === 0 ? secondFrame : firstFrame
            buffer.load(upcoming)
        }
    }

    Timer { id: refresh; interval: 0; onTriggered: asciiRoot.updateFrame() }
    Connections {
        target: asciiRoot.Window.window
        function onSceneGraphInvalidated() {
            asciiRoot.cachedDescriptor = null
            asciiRoot.pendingDescriptor = null
            asciiRoot.outputDirty = true
        }
        function onSceneGraphInitialized() { refresh.restart() }
    }
    Item {
        id: shaderSource
        anchors.fill: parent
        AsciiFrame {
            id: firstFrame
            anchors.fill: parent
            active: asciiRoot.activeBuffer === 0
            onCompleted: asciiRoot.present(firstFrame)
            onLoadFailed: key => { if (key === asciiRoot.desiredKey) asciiRoot.frameError = "Could not load a prepared frame." }
        }
        AsciiFrame {
            id: secondFrame
            anchors.fill: parent
            active: asciiRoot.activeBuffer === 1
            onCompleted: asciiRoot.present(secondFrame)
            onLoadFailed: key => { if (key === asciiRoot.desiredKey) asciiRoot.frameError = "Could not load a prepared frame." }
        }
    }
    ShaderEffectSource {
        id: composedFrame
        objectName: "composedFrame"
        anchors.fill: parent
        sourceItem: shaderSource
        textureSize: asciiRoot.outputTextureSize
        format: ShaderEffectSource.RGBA8
        hideSource: true
        live: false
        smooth: asciiRoot.smoothOutput
        mipmap: asciiRoot.mipmapOutput
        samples: 0
        visible: asciiRoot.ready || !!asciiRoot.pendingDescriptor
        onScheduledUpdateCompleted: asciiRoot.finishCapture()
    }
}
