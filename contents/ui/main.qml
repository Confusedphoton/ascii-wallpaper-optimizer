/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick
import org.kde.plasma.plasmoid
import org.kde.plasma.wallpapers.image

WallpaperItem {
    id: root

    readonly property string mode: wallpaper.configuration.Mode || "Image"
    readonly property var configImages: wallpaper.configuration.Images || []

    property int currentIdx: 0
    property int nextIdx: 0

    function getImage(index) {
        if (!configImages || configImages.length === undefined || configImages.length === 0 || index >= configImages.length) {
            return "";
        }

        let raw = String(configImages[index] || "").trim();
        return raw !== "" ? "file://" + raw : "";
    }

    Timer {
        id: intervalTimer
        interval: 5000
        running: root.mode === "Slideshow" && root.configImages.length > 1
        repeat: true
        onTriggered: {
            root.nextIdx = (root.currentIdx + 1) % root.configImages.length;
            crossfadeAnimation.start();
        }
    }

    NumberAnimation {
        id: crossfadeAnimation
        target: shaderContainer
        property: "transitionProgress"
        from: 0.0
        to: 1.0
        duration: 1000
        easing.type: Easing.InOutQuad
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
        
        source: root.getImage(root.currentIdx)
        sourceNext: root.getImage(root.nextIdx)
        
        detail: (wallpaper.configuration.Detail !== undefined) ? wallpaper.configuration.Detail : 1
        grayscale: wallpaper.configuration.Grayscale || false
    }

    Component.onCompleted: {
        root.loading = false;
    }
}
