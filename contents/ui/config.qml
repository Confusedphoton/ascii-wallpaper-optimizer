/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import org.kde.kirigami as Kirigami

Kirigami.ScrollablePage {
    id: root
    property var configDialog
    property var wallpaperConfiguration

    property alias cfg_Image: pathLabel.text
    property alias cfg_Detail: detailSlider.value
    property alias cfg_Grayscale: grayscaleChecker.checked

    FileDialog {
        id: imagePicker
        title: "Select Wallpaper"
        nameFilters: ["Image files (*.png *.jpg *.jpeg)"]
        onAccepted: pathLabel.text = selectedFile.toString()
    }

    
    // --- Preview ---
    ColumnLayout {
        anchors.left: parent.left
        anchors.right: parent.right
        spacing: Kirigami.Units.largeSpacing

        Kirigami.Card {
    id: previewCard
    Layout.alignment: Qt.AlignHCenter
    Layout.fillWidth: true
    Layout.maximumWidth: Kirigami.Units.gridUnit * 25
    
    Layout.preferredHeight: (width * 9) / 16
    
    contentItem: Item {
        id: previewContainer
        clip: true

        Item {
        id: offscreenBuffer
        width: 1280
        height: 720
        visible: false

        AsciiShader {
            anchors.fill: parent
            source: pathLabel.text
            detail: detailSlider.value
            grayscale: grayscaleChecker.checked
            visible: pathLabel.text !== ""
        }
    }

    ShaderEffectSource {
        id: previewImage
        anchors.fill: parent
        sourceItem: offscreenBuffer

        textureSize: Qt.size(1280, 720)
        samples: 5
        mipmap: true
        smooth: true
        
        live: true 
        hideSource: true
    }

        Controls.Label {
            anchors.centerIn: parent
            text: "No Image Selected"
            opacity: 0.5
            visible: !pathLabel.text
        }

        Rectangle {
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: Kirigami.Units.smallSpacing
            color: Qt.rgba(0, 0, 0, 0.6)
            radius: 4
            width: previewText.width + (Kirigami.Units.smallSpacing * 2)
            height: previewText.height + Kirigami.Units.smallSpacing

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

            RowLayout {
                Kirigami.FormData.label: "Wallpaper:"
                
                Controls.TextField {
                    id: pathDisplay
                    text: pathLabel.text.replace(/^(file:\/{2})/,"")
                    placeholderText: "Select a file..."
                    readOnly: true
                    Layout.fillWidth: true
                }
                
                Controls.Button {
                    icon.name: "document-open"
                    onClicked: imagePicker.open()
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
                id: grayscaleChecker
                Kirigami.FormData.label: "Visual Style:"
                text: "Enable Grayscale effect"
            }
        }

        Controls.Label {
            id: pathLabel
            visible: false
        }
        
        Item { Layout.fillHeight: true }
    }
}
