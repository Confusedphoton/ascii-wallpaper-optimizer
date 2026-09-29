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

    property string overrideProfile: "auto"
    property int timeRevision: 0
    readonly property string scheduledProfile: {
        timeRevision
        if (!wallpaper.configuration.ScheduleEnabled)
            return wallpaper.configuration.ManualProfile === "B" ? "B" : "A"
        const parse = value => {
            const parts = String(value).split(":").map(Number)
            return parts.length === 2 && parts[0] >= 0 && parts[0] < 24 && parts[1] >= 0 && parts[1] < 60
                ? parts[0] * 60 + parts[1] : 0
        }
        const a = parse(wallpaper.configuration.ProfileAStart || "07:00")
        const b = parse(wallpaper.configuration.ProfileBStart || "19:00")
        if (a === b) return "A"
        const now = new Date()
        const minute = now.getHours() * 60 + now.getMinutes()
        return a <= b ? (minute >= a && minute < b ? "A" : "B")
                      : (minute >= a || minute < b ? "A" : "B")
    }
    readonly property string activeProfile: overrideProfile === "auto" ? scheduledProfile : overrideProfile
    readonly property string mode: activeProfile === "A" ? (wallpaper.configuration.Mode || "Image")
        : (wallpaper.configuration.ModeB || "Image")
    readonly property var configImages: activeProfile === "A" ? (wallpaper.configuration.Images || [])
        : (wallpaper.configuration.ImagesB || [])
    readonly property int slideInterval: {
        let v = (wallpaper.configuration.SlideInterval !== undefined) ? wallpaper.configuration.SlideInterval : 10;
        return Math.max(5, v);
    }

    readonly property int transitionFrames: wallpaper.configuration.TransitionFrames || 16
    readonly property int transitionDuration: Math.max(100, wallpaper.configuration.TransitionDuration || 1500)
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
        transitionPace: wallpaper.configuration.TransitionPace || "ease-in-out"
        transitionField: wallpaper.configuration.TransitionField || "blue-noise"
        transitionSpread: (wallpaper.configuration.TransitionSpread !== undefined)
            ? wallpaper.configuration.TransitionSpread : 0.65
        dither: (wallpaper.configuration.Dither !== undefined) ? wallpaper.configuration.Dither : 0.005
        gradientLossStrength: (wallpaper.configuration.RenderStyle || "headroom") === "headroom"
            ? ((wallpaper.configuration.GradientLossStrength !== undefined)
                ? wallpaper.configuration.GradientLossStrength : 1.5) : 0
        structureDither: (wallpaper.configuration.StructureDither !== undefined)
            ? wallpaper.configuration.StructureDither : 0.06
        renderStyle: wallpaper.configuration.RenderStyle || "headroom"
        invertInk: !!wallpaper.configuration.InvertInk
        background: wallpaper.configuration.BackgroundMode === 1
            ? (wallpaper.configuration.BackgroundColor || "#202028") : "auto"
        cachedDirectory: wallpaper.configuration.RenderDirectory || ""
        onRenderStateChanged: if (renderState === "ready") root.renderRevision++
    }

    Timer {
        id: intervalTimer

        running: root.mode === "Slideshow" && root.configImages.length > 1
            && root.currentIdx === root.nextIdx && shaderContainer.ready
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

        duration: root.transitionDuration
        easing.type: Easing.Linear

        onFinished: {
            root.currentIdx = root.nextIdx;
            shaderContainer.transitionProgress = 0.0;
        }
    }

    onConfigImagesChanged: resetSlideshow()
    onModeChanged: resetSlideshow()
    onActiveProfileChanged: resetSlideshow()

    Timer {
        interval: 2000
        repeat: true
        running: true
        onTriggered: {
            root.timeRevision++
            root.fetchOverride()
        }
    }

    function fetchOverride() {
        const xhr = new XMLHttpRequest()
        xhr.open("GET", "http://127.0.0.1:48741/v1/override", true)
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                try {
                    const value = JSON.parse(xhr.responseText).profile
                    if (["auto", "A", "B"].includes(value)) root.overrideProfile = value
                } catch (failure) { /* keep the last known override */ }
            }
        }
        xhr.send()
    }

    function resetSlideshow() {
        crossfadeAnimation.stop();
        shaderContainer.transitionProgress = 0.0;
        
        root.currentIdx = 0;
        root.nextIdx = 0;
    }

    AsciiShader {
        id: shaderContainer
        anchors.fill: parent
        
        renderDirectory: renderer.displayDirectory
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
        root.fetchOverride();
    }
}
