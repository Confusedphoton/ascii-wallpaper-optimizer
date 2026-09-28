/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick

Item {
    id: asciiRoot

    property string renderDirectory: ""
    property int currentIndex: 0
    property int nextIndex: 0
    property real transitionProgress: 0.0
    property int transitionFrames: 16
    property int detail: 1
    property int revision: 0

    readonly property bool ready: glyphMap.status === Image.Ready
        && colorMap.status === Image.Ready && atlasImage.status === Image.Ready
        && metadataImage.status === Image.Ready

    readonly property bool transitioning: currentIndex !== nextIndex
    readonly property int frameIndex: Math.min(transitionFrames - 1,
        Math.floor(transitionProgress * (transitionFrames - 1)))

    function fileUrl(relative) {
        const directory = String(renderDirectory || "").trim().replace(/\/$/, "");
        if (!directory) return "";
        return "file://" + (directory + "/" + relative).split("/").map(encodeURIComponent).join("/")
            + "?v=" + revision;
    }

    function mapPath(kind) {
        if (transitioning) {
            const pair = String(currentIndex).padStart(4, "0") + "-" + String(nextIndex).padStart(4, "0");
            const frame = String(frameIndex).padStart(4, "0");
            return fileUrl("transitions/" + pair + "/" + frame + "/" + kind + ".png");
        }
        return fileUrl("stills/" + String(currentIndex).padStart(4, "0") + "/" + kind + ".png");
    }

    Image {
        id: glyphMap
        source: asciiRoot.mapPath("glyph")
        visible: false
        smooth: false
        cache: true
        layer.enabled: true
        asynchronous: true
    }

    Image {
        id: colorMap
        source: asciiRoot.mapPath("color")
        visible: false
        smooth: false
        cache: true
        layer.enabled: true
        asynchronous: true
    }

    Image {
        id: atlasImage
        source: asciiRoot.fileUrl("atlas.png")
        visible: false
        smooth: false
        cache: true
        layer.enabled: true
        asynchronous: true
    }

    Image {
        id: metadataImage
        source: asciiRoot.fileUrl("metadata.png")
        visible: false
        smooth: false
        cache: true
        layer.enabled: true
        asynchronous: true
    }

    ShaderEffect {
        anchors.fill: parent
        visible: asciiRoot.ready
        blending: false
        property variant glyphs: glyphMap
        property variant colors: colorMap
        property variant atlas: atlasImage
        property variant metadata: metadataImage
        property size gridSize: Qt.size(glyphMap.implicitWidth, glyphMap.implicitHeight)
        vertexShader: "shaders/ascii.vert.qsb"
        fragmentShader: "shaders/ascii.frag.qsb"
    }
}
