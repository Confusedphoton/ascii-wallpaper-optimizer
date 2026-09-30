/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick
import "RenderSettings.js" as RenderSettings
import QtQuick.Controls as Controls
import org.kde.plasma.plasmoid

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
    readonly property var activeSettings: {
        const config = wallpaper.configuration
        const b = activeProfile === "B"
        const useBSettings = b && !!config.ProfileSettingsInitialized
        const setting = name => config[name + (useBSettings ? "B" : "")]
        let chars = Array.from(setting("AsciiChars") || [" ", ".", "´", ":", "r", "s", "z", "u", "n", "v", "c", "x", "L", "O", "G", "D", "Q", "0", "K", "N", "M", "W", "&", "#", "\\", "/", "=", "^", "|", "[", "]", "{", "}"])
        if (chars.length && chars[0] === "\\s") chars[0] = " "
        const style = setting("RenderStyle") || "headroom"
        const currentChars = chars.join("")
        const standardChars = " .´:rszunvcxLOGDQ0KNMW&#\\/=^|[]{}"
        const structureChars = " .´:-_/\\=^|+rszunvcxLOGDQ0KNMW&#[]{}"
        const previousDefaults = [
            " .´:rszunvcxLOGDQ0KNMW&@#\\/=^|[]{}",
            " .´:rszunvcxLOGDQ0KNMW&@\\/=^|[]{}",
            " .´:rszunvcxLOGDQ0KNMW&@\\=^|[]{}",
            " .´:rszunvcxLOGDQ0KNMW&@",
            " .´:rszunvcxLOGDQ0KNMW&@\\=^",
            " .´:-_/\\=^|+rszunvcxLOGDQ0KNMW&@[]{}",
            " .´:-_/|+rszunvcxLOGDQ0KNMW&@[]{}"
        ]
        if (previousDefaults.includes(currentChars)
                || (style === "structure" && currentChars === standardChars)
                || (style === "headroom" && currentChars === structureChars))
            chars = Array.from(style === "structure" ? structureChars : standardChars)
        return {
            fontFamily: setting("FontFamily") || "Monospace",
            characters: chars.map((glyph, index) => index === 0 && glyph === "\\s" ? " " : glyph).join(""),
            detail: setting("Detail") !== undefined ? setting("Detail") : 1,
            grayscale: !!setting("Grayscale"),
            slideInterval: Math.max(5, setting("SlideInterval") !== undefined ? setting("SlideInterval") : 10),
            transitionFrames: setting("TransitionFrames") || 16,
            transitionDuration: Math.max(100, setting("TransitionDuration") || 1500),
            transitionPace: setting("TransitionPace") || "ease-in-out",
            transitionField: setting("TransitionField") || "blue-noise",
            transitionSpread: setting("TransitionSpread") !== undefined ? setting("TransitionSpread") : 0.65,
            dither: setting("Dither") !== undefined ? setting("Dither") : 0.005,
            gradientLossStrength: setting("GradientLossStrength") !== undefined ? setting("GradientLossStrength") : 1.5,
            glyphLuminanceBonus: setting("GlyphLuminanceBonus") !== undefined ? setting("GlyphLuminanceBonus") : 0.12,
            structureDither: setting("StructureDither") !== undefined ? setting("StructureDither") : 0.06,
            renderStyle: style,
            invertInk: !!setting("InvertInk"),
            backgroundMode: setting("BackgroundMode") !== undefined ? setting("BackgroundMode") : 0,
            backgroundColor: setting("BackgroundColor") || "#202028",
            renderDirectory: b && !useBSettings ? "" : (setting("RenderDirectory") || ""),
            renderedState: b && !useBSettings ? "" : (setting("RenderedState") || "")
        }
    }
    property var overrideRequest: null

    RenderClient {
        id: renderer
        requestSettings: RenderSettings.request(
            root.mode === "Slideshow" ? Array.from(root.configImages) : Array.from(root.configImages).slice(0, 1), root.activeSettings)
        cachedDirectory: root.activeSettings.renderDirectory
        cachedState: root.activeSettings.renderedState
        playbackSettings: ({mode: root.mode, slideInterval: root.activeSettings.slideInterval,
            transitionDuration: root.activeSettings.transitionDuration})
        onAssetsReady: {
            const name = root.activeProfile === "B" ? "RenderDirectoryB" : "RenderDirectory"
            if (wallpaper.configuration[name] !== displayDirectory)
                wallpaper.configuration[name] = displayDirectory
            const stateName = root.activeProfile === "B" ? "RenderedStateB" : "RenderedState"
            if (wallpaper.configuration[stateName] !== publishedState)
                wallpaper.configuration[stateName] = publishedState
            wallpaper.configuration.writeConfig()
        }
    }

    SlideshowPlayer {
        id: slideshow
        ready: shaderContainer.ready
        loading: shaderContainer.loading
        enabled: renderer.displayPlaybackSettings.mode === "Slideshow"
        imageCount: renderer.displayImageCount
        frameCount: renderer.displayFrameCount
        generation: renderer.displayGeneration
        slideInterval: (renderer.displayPlaybackSettings.slideInterval || 10) * 1000
        transitionDuration: renderer.displayPlaybackSettings.transitionDuration || 1500
    }

    Timer {
        interval: 2000
        repeat: true
        running: true
        onTriggered: {
            root.timeRevision++
            root.fetchOverride()
        }
    }

    Timer {
        id: overrideTimeout
        interval: 5000
        onTriggered: {
            const previous = root.overrideRequest
            root.overrideRequest = null
            if (previous) previous.abort()
        }
    }

    function fetchOverride() {
        if (overrideRequest) return
        const xhr = new XMLHttpRequest()
        overrideRequest = xhr
        xhr.open("GET", "http://127.0.0.1:48741/v1/override", true)
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || root.overrideRequest !== xhr) return
            root.overrideRequest = null
            overrideTimeout.stop()
            if (xhr.status === 200) {
                try {
                    const value = JSON.parse(xhr.responseText).profile
                    if (["auto", "A", "B"].includes(value)) root.overrideProfile = value
                } catch (failure) { /* keep the last known override */ }
            }
        }
        overrideTimeout.restart()
        xhr.send()
    }

    AsciiShader {
        id: shaderContainer
        anchors.fill: parent
        
        renderDirectory: renderer.displayDirectory
        generation: renderer.displayGeneration
        manifest: renderer.displayManifest
        imageCount: renderer.displayImageCount
        currentIndex: slideshow.currentIndex
        nextIndex: slideshow.nextIndex
        transitionProgress: slideshow.transitionProgress
        transitionFrame: slideshow.transitionFrame
        transitionFrames: renderer.displayFrameCount
    }

    Rectangle {
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20
        width: Math.min(480, root.width - 40)
        height: statusColumn.implicitHeight + 24
        radius: 8
        color: "#b0000000"
        visible: !shaderContainer.ready || renderer.renderState === "queued" || renderer.renderState === "running"
            || (renderer.renderState === "error" && !renderer.retrying)

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

    Component.onDestruction: {
        const previous = overrideRequest
        overrideRequest = null
        if (previous) previous.abort()
    }

    Component.onCompleted: {
        root.loading = false;
        root.fetchOverride();
    }
}
