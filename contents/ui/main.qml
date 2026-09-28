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
    readonly property int slideInterval: {
        let v = (wallpaper.configuration.SlideInterval !== undefined) ? wallpaper.configuration.SlideInterval : 10;
        return Math.max(5, v);
    }

    readonly property string renderDirectory: wallpaper.configuration.RenderDirectory || ""
    readonly property int transitionFrames: wallpaper.configuration.TransitionFrames || 16

    property int currentIdx: 0
    property int nextIdx: 0

    Timer {
        id: intervalTimer

        running: root.mode === "Slideshow" && root.configImages.length > 1
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
        
        renderDirectory: root.renderDirectory
        currentIndex: root.currentIdx
        nextIndex: root.nextIdx
        transitionFrames: root.transitionFrames
        detail: (wallpaper.configuration.Detail !== undefined) ? wallpaper.configuration.Detail : 1
    }

    Component.onCompleted: {
        root.loading = false;
    }
}
