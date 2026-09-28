/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick
import QtQuick.Controls as Controls
import org.kde.plasma.plasmoid
import org.kde.plasma.wallpapers.image

WallpaperItem {
    id: root

    readonly property string mode: wallpaper.configuration.Mode || "Image"
    readonly property var configImages: wallpaper.configuration.Images || []
    readonly property int slideInterval: {
        let v = (wallpaper.configuration.SlideInterval !== undefined) ? wallpaper.configuration.SlideInterval : 10;
        return Math.max(5, v);
    }

    readonly property int transitionFrames: wallpaper.configuration.TransitionFrames || 16
    property int renderRevision: 0

    property int currentIdx: 0
    property int nextIdx: 0

    RenderClient {
        id: renderer
        images: root.mode === "Slideshow" ? Array.from(root.configImages)
            : Array.from(root.configImages).slice(0, 1)
        fontFamily: wallpaper.configuration.FontFamily || "Monospace"
        characters: Array.from(wallpaper.configuration.AsciiChars || []).map(
            (glyph, index) => index === 0 && glyph === "\\s" ? " " : glyph).join("")
        detail: (wallpaper.configuration.Detail !== undefined) ? wallpaper.configuration.Detail : 1
        grayscale: !!wallpaper.configuration.Grayscale
        frames: root.transitionFrames
        dither: (wallpaper.configuration.Dither !== undefined) ? wallpaper.configuration.Dither : 0.005
        onRenderStateChanged: if (renderState === "ready") root.renderRevision++
    }

    Image {
        anchors.fill: parent
        source: root.configImages.length > 0 ? "file://" + String(root.configImages[0]).split("/").map(encodeURIComponent).join("/") : ""
        sourceSize: Qt.size(Math.max(1, root.width), Math.max(1, root.height))
        fillMode: Image.PreserveAspectCrop
        asynchronous: true
        visible: !shaderContainer.ready
    }

    Timer {
        id: intervalTimer

        running: root.mode === "Slideshow" && root.configImages.length > 1 && shaderContainer.ready
        interval: root.slideInterval * 1000
        repeat: true

        onTriggered: {
            root.nextIdx = (root.currentIdx + 1) % root.configImages.length;
            crossfadeAnimation.start();
        }
    }

    NumberAnimation {
        id: crossfadeAnimation

        property: "transitionProgress"
        target: shaderContainer

        from: 0.0
        to: 1.0

        duration: 1000
        easing.type: Easing.Linear

        onFinished: {
            root.currentIdx = root.nextIdx;
            shaderContainer.transitionProgress = 0.0;
        }
    }

    onConfigImagesChanged: resetSlideshow()
    onModeChanged: resetSlideshow()

    function resetSlideshow() {
        crossfadeAnimation.stop();
        shaderContainer.transitionProgress = 0.0;
        
        root.currentIdx = 0;
        root.nextIdx = 0;
    }

    AsciiShader {
        id: shaderContainer
        anchors.fill: parent
        
        renderDirectory: renderer.renderState === "ready" ? renderer.directory : ""
        currentIndex: root.currentIdx
        nextIndex: root.nextIdx
        transitionFrames: root.transitionFrames
        detail: (wallpaper.configuration.Detail !== undefined) ? wallpaper.configuration.Detail : 1
        revision: root.renderRevision
    }

    Rectangle {
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20
        width: Math.min(480, root.width - 40)
        height: statusColumn.implicitHeight + 24
        radius: 8
        color: "#b0000000"
        visible: renderer.renderState !== "ready" || !shaderContainer.ready

        Column {
            id: statusColumn
            anchors.fill: parent
            anchors.margins: 12
            spacing: 6
            Controls.Label {
                width: parent.width
                wrapMode: Text.WordWrap
                color: "white"
                text: renderer.renderState === "error" ? renderer.error
                    : renderer.renderState === "idle" ? "Select an image in wallpaper settings"
                    : renderer.stage || "Preparing ASCII wallpaper"
            }
            Controls.ProgressBar {
                width: parent.width
                visible: renderer.renderState === "running" || renderer.renderState === "queued" || renderer.renderState === "connecting"
                from: 0
                to: 1
                value: renderer.progress
                indeterminate: renderer.renderState === "connecting" || (renderer.renderState === "running" && renderer.progress === 0)
            }
        }
    }

    Component.onCompleted: {
        root.loading = false;
    }
}
