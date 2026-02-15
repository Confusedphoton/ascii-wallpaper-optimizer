/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick
import org.kde.plasma.plasmoid
import org.kde.plasma.wallpapers.image

WallpaperItem {
    id: root

    AsciiShader {
        anchors.fill: parent
        source: wallpaper.configuration.Image ? wallpaper.configuration.Image : ""
        detail: (wallpaper.configuration.Detail !== undefined) ? wallpaper.configuration.Detail : 1
        grayscale: wallpaper.configuration.Grayscale || false
    }

    Component.onCompleted: {
        root.loading = false;
    }
}
