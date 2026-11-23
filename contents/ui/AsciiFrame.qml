/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick

Item {
    id: frame
    property var descriptor: null
    property bool active: false
    property bool loadingSources: false
    readonly property string frameKey: descriptor ? descriptor.key : ""
    readonly property var manifest: descriptor ? descriptor.manifest || ({}) : ({})
    readonly property bool hasMetadata: (manifest.output_size || []).length === 2
        && (manifest.cell_size || []).length === 2 && Number(manifest.atlas_count) > 0
    readonly property real sdfSpread: Math.max(0, Number(manifest.sdf_spread) || 0)
    readonly property var frameBackground: (manifest.background_labs || ({}))[descriptor ? descriptor.path : ""]
    readonly property bool ready: !!descriptor && glyphMap.status === Image.Ready
        && colorMap.status === Image.Ready && atlasImage.status === Image.Ready
        && (hasMetadata || metadataImage.status === Image.Ready) && backgroundMap.status === Image.Ready
    readonly property bool failed: glyphMap.status === Image.Error || colorMap.status === Image.Error
        || atlasImage.status === Image.Error || (!hasMetadata && metadataImage.status === Image.Error)
        || backgroundMap.status === Image.Error
    signal completed()
    signal loadFailed(string key)

    function load(value) {
        if (!value) return
        if (value.key === frameKey) { Qt.callLater(reportReady); return }
        loadingSources = true
        descriptor = value
        const base = value.url + "/"
        const suffix = "?v=" + encodeURIComponent(value.generation)
        glyphMap.source = base + value.path + "/glyph.png" + suffix
        colorMap.source = base + value.path + "/color.png" + suffix
        backgroundMap.source = base + value.path + "/background.png" + suffix
        atlasImage.source = base + "atlas.png" + suffix
        metadataImage.source = base + "metadata.png" + suffix
        loadingSources = false
        Qt.callLater(reportReady)
    }

    function reportReady() {
        if (!loadingSources && ready) completed()
        else if (!loadingSources && failed) loadFailed(frameKey)
    }
    onReadyChanged: reportReady()
    onFailedChanged: reportReady()

    Image { id: glyphMap; visible: false; smooth: false; cache: true; asynchronous: true }
    Image { id: colorMap; visible: false; smooth: false; cache: true; asynchronous: true }
    Image { id: backgroundMap; visible: false; smooth: false; cache: true; asynchronous: true }
    Image { id: atlasImage; visible: false; smooth: frame.sdfSpread > 0; cache: true; asynchronous: true }
    Image { id: metadataImage; visible: false; smooth: false; cache: true; asynchronous: true }

    ShaderEffect {
        anchors.fill: parent
        visible: frame.active && frame.ready
        blending: false
        property variant glyphs: glyphMap
        property variant colors: colorMap
        property variant backgrounds: backgroundMap
        property variant atlas: atlasImage
        property variant metadata: metadataImage
        property size gridSize: Qt.size(glyphMap.implicitWidth, glyphMap.implicitHeight)
        property size outputSize: frame.hasMetadata ? Qt.size(frame.manifest.output_size[0], frame.manifest.output_size[1]) : Qt.size(0, 0)
        property size cellSize: frame.hasMetadata ? Qt.size(frame.manifest.cell_size[0], frame.manifest.cell_size[1]) : Qt.size(0, 0)
        property real atlasCount: frame.hasMetadata ? Number(frame.manifest.atlas_count) : 0
        property real invertInk: frame.manifest.invert_ink ? 1 : 0
        property real metadataAvailable: frame.hasMetadata ? 1 : 0
        property vector3d backgroundLab: frame.frameBackground ? Qt.vector3d(frame.frameBackground[0], frame.frameBackground[1], frame.frameBackground[2]) : Qt.vector3d(0, 0, 0)
        property real backgroundAvailable: frame.frameBackground ? 1 : 0
        property real sdfSpread: frame.sdfSpread
        vertexShader: "shaders/ascii.vert.qsb"
        fragmentShader: "shaders/ascii.frag.qsb"
    }
}
