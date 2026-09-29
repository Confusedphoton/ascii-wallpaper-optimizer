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
    readonly property var activeSettings: {
        const config = wallpaper.configuration
        const b = activeProfile === "B"
        const useBSettings = b && !!config.ProfileSettingsInitialized
        const setting = name => config[name + (useBSettings ? "B" : "")]
        const chars = setting("AsciiChars") || [" ", ".", "´", ":", "r", "s", "z", "u", "n", "v", "c", "x", "L", "O", "G", "D", "Q", "0", "K", "N", "M", "W", "&", "@", "\\", "/", "=", "^", "|", "[", "]", "{", "}"]
        return {
            fontFamily: setting("FontFamily") || "Monospace",
            characters: Array.from(chars).map((glyph, index) => index === 0 && glyph === "\\s" ? " " : glyph).join(""),
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
            renderStyle: setting("RenderStyle") || "headroom",
            invertInk: !!setting("InvertInk"),
            backgroundMode: setting("BackgroundMode") !== undefined ? setting("BackgroundMode") : 0,
            backgroundColor: setting("BackgroundColor") || "#202028",
            renderDirectory: b && !useBSettings ? "" : (setting("RenderDirectory") || "")
        }
    }
    property int renderRevision: 0
    property int transitionRevision: 0

    property int currentIdx: 0
    property int nextIdx: 0

    RenderClient {
        id: renderer
        requestSettings: ({
            images: root.mode === "Slideshow" ? Array.from(root.configImages)
                : Array.from(root.configImages).slice(0, 1),
            font: root.activeSettings.fontFamily,
            chars: root.activeSettings.characters,
            detail: root.activeSettings.detail,
            grayscale: root.activeSettings.grayscale,
            frames: root.activeSettings.transitionFrames,
            transition_pace: root.activeSettings.transitionPace,
            transition_field: root.activeSettings.transitionField,
            transition_spread: root.activeSettings.transitionSpread,
            dither: root.activeSettings.dither,
            glyph_luminance_bonus: root.activeSettings.renderStyle === "headroom" ? root.activeSettings.glyphLuminanceBonus : 0,
            gradient_loss_strength: root.activeSettings.renderStyle === "headroom" ? root.activeSettings.gradientLossStrength : 0,
            structure_dither: root.activeSettings.structureDither,
            style: root.activeSettings.renderStyle,
            invert_ink: root.activeSettings.invertInk,
            background: root.activeSettings.backgroundMode === 1 ? root.activeSettings.backgroundColor : "auto"
        })
        cachedDirectory: root.activeSettings.renderDirectory
        onRenderStateChanged: {
            if (renderState === "ready") {
                root.transitionRevision++
                if (currentRequestForced) root.renderRevision++
            }
        }
    }

    Timer {
        id: intervalTimer

        running: root.mode === "Slideshow" && root.configImages.length > 1
            && root.currentIdx === root.nextIdx && shaderContainer.ready
        interval: root.activeSettings.slideInterval * 1000
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

        duration: root.activeSettings.transitionDuration
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
        transitionFrames: root.activeSettings.transitionFrames
        detail: root.activeSettings.detail
        revision: root.renderRevision
        transitionRevision: root.transitionRevision
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
