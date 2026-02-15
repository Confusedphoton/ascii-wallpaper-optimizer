/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick
import QtQuick.Controls

Item {
    id: asciiRoot
    property url source: ""
    property int detail: 1
    property bool grayscale: false

    readonly property int charWidth: [2, 4, 8][detail]
    readonly property int charHeight: charWidth * 2
    
    property var asciiChars: [" ", ".", "´", ":", "r", "s", "z", "u", "n", "v", "c", "x", "L", "O", "G", "D", "Q", "0", "K", "N", "M", "W", "&", "@"]

    Image {
        id: sourceImg
        source: asciiRoot.source
        visible: false
        asynchronous: true
    }

    Canvas {
        id: atlasGenerator
        
        width: asciiChars.length * (charWidth * 2)
        height: charHeight

        visible: false
        renderTarget: Canvas.FramebufferObject

        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()

        onPaint: {
            var ctx = getContext("2d");
            ctx.imageSmoothingEnabled = true;

            ctx.reset();
            ctx.clearRect(0, 0, width, height);

            ctx.font = charHeight + "px monospace";
            ctx.textBaseline = "middle";
            
            ctx.textAlign = "center";
            ctx.fillStyle = "white";

            let slotWidth = charWidth * 2;
            for (var i = 0; i < asciiChars.length; i++) {
                let xCenter = (i * slotWidth) + (slotWidth / 2);
                ctx.fillText(asciiChars[i], xCenter, charHeight / 2);
            }
        }
        onAvailableChanged: if (available) requestPaint()
        onVisibleChanged: if (visible) requestPaint()
        Component.onCompleted: requestPaint()
    }

    ShaderEffect {
        anchors.fill: parent
        
        property variant src: sourceImg
        property variant atlas: atlasGenerator 

        property real invAtlasCount: 1.0 / asciiChars.length

        property size charSize: Qt.size(charWidth, charHeight)
        property size viewSize: Qt.size(width, height)
        
        property int atlasCount: asciiChars.length
        property int isGrayscaled: asciiRoot.grayscale ? 1 : 0

        vertexShader: "shaders/ascii.vert.qsb"
        fragmentShader: "shaders/ascii.frag.qsb"
    }
}
