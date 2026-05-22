/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtCore
import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Kirigami.ScrollablePage {
    id: root

    property var configDialog
    property var wallpaperConfiguration

    property list<string> cfg_Images: []
    property string cfg_Mode: "Image"
    property alias cfg_Detail: detailSlider.value
    property alias cfg_Grayscale: grayscaleCheck.checked

    ListModel { id: imageList }
    property int _activePicker: -1

    function syncToConfig() {
        const unique = new Set();

        for (let i = 0; i < imageList.count; ++i) {
            const raw = imageList.get(i).path;
            if (!raw) continue;

            const clean = cleanPath(raw);
            if (clean) {
                unique.add(clean);
            }
        }
        
        root.cfg_Images = Array.from(unique);
        if (root.configDialog) {
            root.configDialog.changed = true;
        }
    }

    function clearImages() {
        imageList.clear()

        if (root.cfg_Mode !== "Slideshow") {
            imageList.append({ "path": "" })
        }

        root.syncToConfig()
        previewContainer.resetPreview()
    }

    function removeImage(index) {
        if (index >= 0 && index < imageList.count) {
            imageList.remove(index);

            syncToConfig();
            previewContainer.resetPreview()
        }
    }

    function cleanPath(url) {
        return url.replace(/^file:\/{2}/,"").trim();
    }

    Component.onCompleted: {
        imageList.clear();

        let currentImages = root.cfg_Images || [];
        for (let i = 0; i < currentImages.length; i++) {
            if (String(currentImages[i]).trim() !== "") {
                imageList.append({ "path": String(currentImages[i]).trim() });
            }
        }
    }

    FileDialog {
        id: imagePicker
        title: "Select Wallpaper"

        currentFolder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0]
        fileMode: root.cfg_Mode === "Slideshow" ? FileDialog.OpenFiles : FileDialog.OpenFile
        nameFilters: ["Image files (*.png *.jpg *.jpeg)"]

        onAccepted: {
            if (fileMode === FileDialog.OpenFiles) {
                for (const file of selectedFiles) {
                    const clean = cleanPath(file.toString());

                    if (!clean) continue;

                    let exists = false;
                    for (let i = 0; i < imageList.count; ++i) {
                        if (cleanPath(imageList.get(i).path) === clean) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        imageList.append({ "path": file.toString() });
                    }
                }
            } else {
                if (_activePicker !== -1) {
                    const clean = cleanPath(selectedFile.toString());

                    if (_activePicker === 0 && imageList.count > 1) {
                        imageList.remove(1, imageList.count - 1)
                    }

                    let duplicate = false
                    for (let i = 0; i < imageList.count; ++i) {
                        if (i === _activePicker) continue
                        if (cleanPath(imageList.get(i).path) === clean) {
                            duplicate = true; break
                        }
                    }

                    if (!duplicate) {
                        imageList.setProperty(_activePicker, "path", selectedFile.toString())
                    }
                }

                _activePicker = -1
            }

            root.syncToConfig();
            previewContainer.resetPreview()
        }
        onRejected: {
            root._activePicker = -1;
        }
    }
    
    ColumnLayout {
        anchors.left: parent.left
        anchors.right: parent.right
        spacing: Kirigami.Units.largeSpacing

        // --- Preview ---
        Kirigami.Card {
            id: previewCard

            Layout.alignment: Qt.AlignHCenter
            Layout.fillWidth: true
            Layout.maximumWidth: Kirigami.Units.gridUnit * 25
            Layout.preferredHeight: (width * 9) / 16
            
            contentItem: Item {
                id: previewContainer
                clip: true

                property int currentIdx: 0
                property int nextIdx: 0


                function resetPreview() {
                    previewTimer.stop();
                    
                    if (currentIdx !== 0 && imageList.count > 0) {
                        crossfadeAnimation.stop();
                        previewShader.transitionProgress = 0.0;

                        nextIdx = 0;
                        crossfadeAnimation.restart();
                    } else {
                        crossfadeAnimation.stop();
                        previewShader.transitionProgress = 0.0;

                        currentIdx = 0;
                        nextIdx = 0;
                    }

                    previewTimer.restart();
                }

                onCurrentIdxChanged: previewShader.transitionProgress = 0.0
                Connections {
                    target: root
                    onCfg_ModeChanged: previewContainer.resetPreview()
                }

                function getPreviewPath(index) {
                    if (imageList.count === 0 || index < 0 || index >= imageList.count) {
                        return "";
                    }

                    let path = imageList.get(index).path;
                    return path.startsWith("file://") ? path : "file://" + path;
                }

                Timer {
                    id: previewTimer
                    
                    interval: 5000
                    running: root.cfg_Mode === "Slideshow" && imageList.count > 1
                    repeat: true

                    onTriggered: {
                        if (imageList.count > 1) {
                            previewContainer.nextIdx = (previewContainer.currentIdx + 1) % imageList.count;
                            crossfadeAnimation.restart();
                        }
                    }
                }

                NumberAnimation {
                    id: crossfadeAnimation

                    property: "transitionProgress"
                    target: previewShader

                    from: 0.0
                    to: 1.0

                    duration: 1500
                    easing.type: Easing.InOutQuad
                    
                    onFinished: {
                        previewContainer.currentIdx = previewContainer.nextIdx;
                        previewShader.transitionProgress = 0.0;
                    }
                }

                Item {
                    id: offscreenBuffer

                    width: 1280
                    height: 720

                    AsciiShader {
                        id: previewShader
                        anchors.fill: parent
                        
                        source: previewContainer.getPreviewPath(previewContainer.currentIdx)
                        sourceNext: previewContainer.getPreviewPath(previewContainer.nextIdx)
                        
                        detail: root.cfg_Detail
                        grayscale: root.cfg_Grayscale
                    }
                }

                ShaderEffectSource {
                    id: previewImage
                    sourceItem: offscreenBuffer
                    
                    anchors.fill: parent
                    anchors.centerIn: parent
                    
                    textureSize: Qt.size(1280, 720)
                    width: parent.width
                    height: parent.height
                    
                    samples: 5
                    mipmap: true
                    smooth: true

                    live: true
                    recursive: true
                    hideSource: true
                }

                Controls.Label {
                    anchors.centerIn: parent
                    text: "No Image Selected"
                    opacity: 0.5
                    visible: previewContainer.getPreviewPath(previewContainer.currentIdx) === ""
                }

                Rectangle {
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.margins: Kirigami.Units.smallSpacing

                    width: previewText.width + (Kirigami.Units.smallSpacing * 2)
                    height: previewText.height + Kirigami.Units.smallSpacing
                    radius: 4
                    color: Qt.rgba(0, 0, 0, 0.6)

                    Controls.Label {
                        id: previewText
                        anchors.centerIn: parent
                        text: "Preview"
                        font.pointSize: 8
                        color: "white"
                        font.weight: Font.Bold
                    }
                }
            }
        }

        // --- Options ---
        Kirigami.FormLayout {
            Layout.fillWidth: true

            Kirigami.Separator {
                Kirigami.FormData.isSection: true
                Kirigami.FormData.label: "Mode"
            }

            Controls.ComboBox {
                id: modeCombo
                Kirigami.FormData.label: "Wallpaper Mode:"
                
                model: ["Image", "Slideshow"]
                currentIndex: model.indexOf(root.cfg_Mode)

                onCurrentTextChanged: {
                    root.cfg_Mode = currentText
        

                    if (root.configDialog) {
                        root.configDialog.changed = true
                    }
                }
            }

            Kirigami.Separator {
                Kirigami.FormData.isSection: true
                Kirigami.FormData.label: "Wallpapers"
            }

            ColumnLayout {
                Kirigami.FormData.isSection: true

                Layout.fillWidth: true
                spacing: Kirigami.Units.smallSpacing
                
                Repeater {
                    model: imageList

                    delegate: RowLayout {
                        Kirigami.FormData.label: "Image " + (index + 1) + ":"

                        Layout.fillWidth: true

                        visible: root.cfg_Mode === "Slideshow" || index === 0

                        ColumnLayout {
                            spacing: -5

                            Layout.preferredHeight: Kirigami.Units.gridUnit * 2
                            Layout.preferredWidth: Kirigami.Units.gridUnit * 1.5

                            Controls.ToolButton {
                                icon.name: "arrow-up"
                                enabled: index > 0
                                
                                Layout.fillWidth: true
                                Layout.fillHeight: true

                                onClicked: {
                                    imageList.move(index, index - 1, 1);

                                    root.syncToConfig();
                                    previewContainer.resetPreview();
                                }
                            }

                            Controls.ToolButton {
                                icon.name: "arrow-down"
                                enabled: index < imageList.count - 1

                                Layout.fillWidth: true
                                Layout.fillHeight: true

                                onClicked: {
                                    imageList.move(index, index + 1, 1);

                                    root.syncToConfig();
                                    previewContainer.resetPreview();
                                }
                            }
                        }

                        Controls.TextField {
                            text: cleanPath(model.path)
                            placeholderText: "Select a file…"
                            readOnly: true
                            Layout.fillWidth: true
                        }

                        Controls.Button {
                            icon.name: "document-open"

                            enabled: root.cfg_Mode !== "Slideshow"
                            visible: root.cfg_Mode === "Slideshow" ? false : true

                            onClicked: {
                                root._activePicker = index;
                                imagePicker.open();
                            }
                        }

                        Controls.Button {
                            icon.name: "list-remove"

                            enabled: imageList.count > 1
                            visible: root.cfg_Mode === "Slideshow" ? true : false
                            
                            onClicked: {
                                root.removeImage(index)
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true

                    visible: root.cfg_Mode === "Slideshow"

                    Controls.Button {
                        Kirigami.FormData.label: ""

                        Layout.fillWidth: true
                        Layout.preferredWidth: 0

                        enabled: root.cfg_Mode === "Slideshow"
                        
                        text: "Select Images"
                        icon.name: "list-add"
                        
                        onClicked: {
                            imagePicker.fileMode = FileDialog.OpenFiles;
                            imagePicker.open();
                        }
                    }

                    Controls.Button {
                        Kirigami.FormData.label: ""

                        Layout.fillWidth: true
                        Layout.preferredWidth: 0

                        enabled: root.cfg_Mode === "Slideshow"
                        
                        text: "Clear Images"
                        icon.name: "edit-delete"
                        
                        onClicked: {
                            root.clearImages();
                        }
                    }
                }
            }

            Kirigami.Separator {
                Kirigami.FormData.isSection: true
                Kirigami.FormData.label: "Adjustments"
            }

            RowLayout {
                Kirigami.FormData.label: "Detail Level:"
                
                Controls.Slider {
                    id: detailSlider
                    from: 2
                    to: 0
                    stepSize: 1
                    snapMode: Controls.Slider.SnapOnRelease
                    Layout.fillWidth: true
                }
                
                Controls.Label {
                    text: {
                        let w = (detailSlider.value === 0) ? 2 : (detailSlider.value === 1 ? 4 : 8);
                        let h = w * 2;
                        let labels = [" (High Detail)", " (Mid Detail)", " (Low Detail)"];
                        return w + "x" + h + labels[detailSlider.value];
                    }
                    font.weight: Font.Bold
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 7
                }
            }

            Controls.CheckBox {
                id: grayscaleCheck
                Kirigami.FormData.label: "Visual Style:"
                text: "Enable Grayscale effect"
            }
        }

        Item { Layout.fillHeight: true }
    }
}
