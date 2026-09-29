/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

pragma ComponentBehavior: Bound

import QtCore
import QtQuick
import "RenderSettings.js" as RenderSettings
import QtQuick.Controls as Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import QtQuick.Effects
import org.kde.kirigami as Kirigami

Item {
    id: root

    implicitWidth: mainLayout.implicitWidth
    implicitHeight: mainLayout.implicitHeight

    property var configDialog
    property var wallpaperConfiguration

    property string cfg_Mode: "Image"
    property string cfg_ModeB: "Image"
    property list<string> cfg_ImagesB: []
    property bool cfg_ProfileSettingsInitialized: false
    property bool cfg_ScheduleEnabled: false
    property string cfg_ManualProfile: "A"
    property string cfg_ProfileAStart: "07:00"
    property string cfg_ProfileBStart: "19:00"
    property int editProfile: 0
    readonly property string editMode: editProfile === 0 ? cfg_Mode : cfg_ModeB
    readonly property var editImages: editProfile === 0 ? cfg_Images : cfg_ImagesB
    property string cfg_FontFamily: "Monospace"
    property string cfg_RenderDirectory: ""
    property int cfg_TransitionFrames: 16
    property int cfg_TransitionDuration: 1500
    property string cfg_TransitionPace: "ease-in-out"
    property string cfg_TransitionField: "blue-noise"
    property real cfg_TransitionSpread: 0.65
    property int cfg_BackgroundMode: 0
    property string cfg_BackgroundColor: "#202028"
    property string cfg_RenderStyle: "headroom"
    property string cfg_FontFamilyB: "Monospace"
    property string cfg_RenderDirectoryB: ""
    property int cfg_TransitionFramesB: 16
    property int cfg_TransitionDurationB: 1500
    property string cfg_TransitionPaceB: "ease-in-out"
    property string cfg_TransitionFieldB: "blue-noise"
    property real cfg_TransitionSpreadB: 0.65
    property int cfg_BackgroundModeB: 0
    property string cfg_BackgroundColorB: "#202028"
    property string cfg_RenderStyleB: "headroom"
    readonly property string standardChars: " .´:rszunvcxLOGDQ0KNMW&@\\/=^|[]{}"
    readonly property string legacyStandardChars: " .´:rszunvcxLOGDQ0KNMW&@\\=^|[]{}"
    readonly property string legacyOlderStandardChars: " .´:rszunvcxLOGDQ0KNMW&@"
    readonly property string structureChars: " .´:-_/\\=^|+rszunvcxLOGDQ0KNMW&@[]{}"
    readonly property string legacyStructureChars: " .´:-_/|+rszunvcxLOGDQ0KNMW&@[]{}"
    property list<string> cfg_AsciiChars: [" ", ".", "´", ":", "r", "s", "z", "u", "n", "v", "c", "x", "L", "O", "G", "D", "Q", "0", "K", "N", "M", "W", "&", "@", "\\", "/", "=", "^", "|", "[", "]", "{", "}"]
    property list<string> cfg_Images: []
    property var monoFonts: [cfg_FontFamily]
    property var fontFamiliesToCheck: []
    property var foundMonoFonts: []
    property int fontScanIndex: 0
    property bool fontScanStarted: false
    property bool controlsReady: false

    function colorHex(color) {
        return "#" + [color.r, color.g, color.b].map(channel =>
            Math.round(channel * 255).toString(16).padStart(2, "0")).join("")
    }

    function applyStyleCharacters(style) {
        const current = Array.from(activeSettings.AsciiChars || []).join("")
        if (style === "structure" && (current === standardChars || current === legacyStandardChars
                || current === legacyOlderStandardChars || current === legacyStructureChars))
            setProfileSetting("AsciiChars", Array.from(structureChars))
        else if (style === "headroom" && (current === legacyStandardChars || current === legacyOlderStandardChars
                || current === structureChars || current === legacyStructureChars))
            setProfileSetting("AsciiChars", Array.from(standardChars))
    }

    property int cfg_Detail: 1
    property int cfg_DetailB: 1
    readonly property var detailValues: [4, 3, 0, 1, 2]
    property bool cfg_Grayscale: false
    property int cfg_SlideInterval: 10
    property real cfg_Dither: 0.005
    property real cfg_GradientLossStrength: 1.5
    property real cfg_GlyphLuminanceBonus: 0.12
    property real cfg_StructureDither: 0.06
    property bool cfg_InvertInk: false
    property list<string> cfg_AsciiCharsB: [" ", ".", "´", ":", "r", "s", "z", "u", "n", "v", "c", "x", "L", "O", "G", "D", "Q", "0", "K", "N", "M", "W", "&", "@", "\\", "/", "=", "^", "|", "[", "]", "{", "}"]
    property bool cfg_GrayscaleB: false
    property int cfg_SlideIntervalB: 10
    property real cfg_DitherB: 0.005
    property real cfg_GradientLossStrengthB: 1.5
    property real cfg_GlyphLuminanceBonusB: 0.12
    property real cfg_StructureDitherB: 0.06
    property bool cfg_InvertInkB: false
    property var activeSettings: ({})

    property string cfg_Image: ""

    ListModel { id: imageList }
    property int _activePicker: -1
    property bool _isInternalUpdate: false

    function makeProfileSettings() {
        const suffix = editProfile === 0 ? "" : "B"
        return {
            FontFamily: root["cfg_FontFamily" + suffix],
            RenderDirectory: root["cfg_RenderDirectory" + suffix],
            TransitionFrames: root["cfg_TransitionFrames" + suffix],
            TransitionDuration: root["cfg_TransitionDuration" + suffix],
            TransitionPace: root["cfg_TransitionPace" + suffix],
            TransitionField: root["cfg_TransitionField" + suffix],
            TransitionSpread: root["cfg_TransitionSpread" + suffix],
            BackgroundMode: root["cfg_BackgroundMode" + suffix],
            BackgroundColor: root["cfg_BackgroundColor" + suffix],
            RenderStyle: root["cfg_RenderStyle" + suffix],
            AsciiChars: root["cfg_AsciiChars" + suffix],
            Detail: root["cfg_Detail" + suffix],
            Grayscale: root["cfg_Grayscale" + suffix],
            SlideInterval: root["cfg_SlideInterval" + suffix],
            Dither: root["cfg_Dither" + suffix],
            GradientLossStrength: root["cfg_GradientLossStrength" + suffix],
            GlyphLuminanceBonus: root["cfg_GlyphLuminanceBonus" + suffix],
            StructureDither: root["cfg_StructureDither" + suffix],
            InvertInk: root["cfg_InvertInk" + suffix]
        }
    }

    function setProfileSetting(name, value) {
        const next = Object.assign({}, activeSettings)
        next[name] = value
        activeSettings = next
        const suffix = editProfile === 0 ? "" : "B"
        root["cfg_" + name + suffix] = value
        if (root.configDialog) root.configDialog.changed = true
    }

    function initializeProfileSettings() {
        if (cfg_ProfileSettingsInitialized) return
        const names = ["FontFamily", "TransitionFrames", "TransitionDuration", "TransitionPace",
            "TransitionField", "TransitionSpread", "BackgroundMode", "BackgroundColor", "RenderStyle",
            "AsciiChars", "Detail", "Grayscale", "SlideInterval", "Dither",
            "GradientLossStrength", "GlyphLuminanceBonus", "StructureDither", "InvertInk"]
        for (const name of names)
            root["cfg_" + name + "B"] = root["cfg_" + name]
        cfg_ProfileSettingsInitialized = true
    }

    onCfg_ImagesChanged: {
        if (editProfile === 0) loadFromConfig();
    }
    onCfg_ImagesBChanged: if (editProfile === 1) loadFromConfig()
    onEditProfileChanged: {
        activeSettings = makeProfileSettings()
        loadFromConfig()
        if (controlsReady) previewContainer.resetPreview()
    }
    onCfg_FontFamilyChanged: {
        if (!monoFonts.includes(cfg_FontFamily))
            monoFonts = [cfg_FontFamily].concat(monoFonts)
        if (controlsReady && editProfile === 0) fontCombo.editText = cfg_FontFamily
    }
    onActiveSettingsChanged: {
        if (!controlsReady) return
        if (!monoFonts.includes(activeSettings.FontFamily))
            monoFonts = [activeSettings.FontFamily].concat(monoFonts)
        fontCombo.editText = activeSettings.FontFamily
    }

    function loadFromConfig() {
        if (_isInternalUpdate) return;
        _isInternalUpdate = true;

        imageList.clear();

        let migrated = false;
        let currentImages = Array.from(root.editImages || []);

        if (editProfile === 0 && root.cfg_Image && root.cfg_Image.trim() !== "") {
            const cleaned = root.cfg_Image.trim();
            const alreadyExists = currentImages.some(p => String(p).trim() === cleaned);

            if (!alreadyExists) {
                currentImages.push(cleaned);
            }

            migrated = true;
            root.cfg_Image = ""; 
            root.cfg_Images = currentImages;
        }

        let currentChars = Array.from(root.activeSettings.AsciiChars || []);
        if (currentChars.length && currentChars[0] === "\\s") {
            currentChars[0] = " ";
            migrated = true;
            root.setProfileSetting("AsciiChars", currentChars);
        }
        if (currentChars.length === 23 && currentChars[0] !== " ") {
            currentChars.unshift(" ");

            migrated = true;
            root.setProfileSetting("AsciiChars", currentChars);
        }

        for (let i = 0; i < currentImages.length; i++) {
            const original = String(currentImages[i]).trim();
            if (original === "") continue;

            const decoded = root.cleanPath(original);
            if (decoded !== original) {
                migrated = true;
            }

            imageList.append({ "path": decoded });
        }

        if (imageList.count === 0 && root.editMode !== "Slideshow") {
            imageList.append({ "path": "" });
        }

        if (migrated) {
            root.syncToConfig();
        }

        _isInternalUpdate = false;
    }

    function syncToConfig() {
        _isInternalUpdate = true;
        const unique = new Set();

        for (let i = 0; i < imageList.count; ++i) {
            const raw = imageList.get(i).path;
            if (!raw) continue;

            const clean = cleanPath(raw);
            if (clean) {
                unique.add(clean);
            }
        }
        
        if (editProfile === 0) root.cfg_Images = Array.from(unique);
        else root.cfg_ImagesB = Array.from(unique);
        if (root.configDialog) {
            root.configDialog.changed = true;
        }
        _isInternalUpdate = false;
    }

    function clearImages() {
        imageList.clear()

        if (root.editMode !== "Slideshow") {
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
        const stripped = String(url).replace(/^file:\/{2}/, "").trim();
        
        try {
            return decodeURIComponent(stripped);
        } catch (e) {
            return stripped;
        }
    }

    function toFileUrl(path) {
        if (!path) return "";
        const clean = String(path).trim();
        if (clean === "") return "";
        return "file://" + clean.split("/").map(encodeURIComponent).join("/");
    }

    Component.onCompleted: {
        initializeProfileSettings()
        activeSettings = makeProfileSettings()
        loadFromConfig()
        applyStyleCharacters(activeSettings.RenderStyle)
        fontCombo.editText = activeSettings.FontFamily
        startFontScan()
        Qt.callLater(function() { root.controlsReady = true })
    }

    function startFontScan() {
        if (fontScanStarted) return
        fontScanStarted = true
        fontFamiliesToCheck = Qt.fontFamilies()
        foundMonoFonts = []
        fontScanIndex = 0
        fontScanTimer.start()
    }

    Timer {
        id: fontScanTimer
        interval: 0
        repeat: true
        onTriggered: {
            let end = Math.min(root.fontScanIndex + 8, root.fontFamiliesToCheck.length)
            for (let i = root.fontScanIndex; i < end; ++i) {
                let family = root.fontFamiliesToCheck[i]
                fontProbe.font = Qt.font({ family: family, pointSize: 12 })
                let widthI = fontProbe.advanceWidth("i")
                if (widthI > 0 && Math.abs(widthI - fontProbe.advanceWidth("W")) < 0.01)
                    root.foundMonoFonts.push(family)
            }
            root.fontScanIndex = end
            if (end >= root.fontFamiliesToCheck.length) {
                stop()
                root.monoFonts = Array.from(new Set([root.activeSettings.FontFamily].concat(root.foundMonoFonts))).sort()
            }
        }
    }

    RenderClient {
        id: renderer
        requestSettings: RenderSettings.request(
            root.editMode === "Slideshow" ? Array.from(root.editImages) : Array.from(root.editImages).slice(0, 1), root.activeSettings)
        cachedDirectory: root.activeSettings.RenderDirectory
        playbackSettings: ({mode: root.editMode, slideInterval: root.activeSettings.SlideInterval,
            transitionDuration: root.activeSettings.TransitionDuration})
        onAssetsReady: {
            if (root.activeSettings.RenderDirectory !== displayDirectory)
                root.setProfileSetting("RenderDirectory", displayDirectory)
        }
    }

    FontMetrics {
        id: fontProbe
    }

    FileDialog {
        id: imagePicker
        title: "Select Wallpaper"

        currentFolder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0]
        fileMode: root.editMode === "Slideshow" ? FileDialog.OpenFiles : FileDialog.OpenFile
        nameFilters: ["Image files (*.png *.jpg *.jpeg)"]

        onAccepted: {
            if (fileMode === FileDialog.OpenFiles) {
                for (const file of selectedFiles) {
                    const clean = root.cleanPath(file.toString());

                    if (!clean) continue;

                    let exists = false;
                    for (let i = 0; i < imageList.count; ++i) {
                        if (root.cleanPath(imageList.get(i).path) === clean) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        imageList.append({ "path": clean });
                    }
                }
            } else {
                if (root._activePicker !== -1) {
                    const clean = root.cleanPath(selectedFile.toString());

                    if (root._activePicker === 0 && imageList.count > 1) {
                        imageList.remove(1, imageList.count - 1)
                    }

                    let duplicate = false
                    for (let i = 0; i < imageList.count; ++i) {
                        if (i === root._activePicker) continue
                        if (root.cleanPath(imageList.get(i).path) === clean) {
                            duplicate = true; break
                        }
                    }

                    if (!duplicate) {
                        imageList.setProperty(root._activePicker, "path", clean)
                    }
                }

                root._activePicker = -1
            }

            root.syncToConfig();
            previewContainer.resetPreview()
        }
        onRejected: {
            root._activePicker = -1;
        }
    }

    ColorDialog {
        id: backgroundPicker
        title: "Select background color"
        selectedColor: root.activeSettings.BackgroundColor
        onAccepted: {
            root.setProfileSetting("BackgroundColor", root.colorHex(selectedColor))
        }
    }
    
    ColumnLayout {
        id: mainLayout
        width: parent.width
        spacing: Kirigami.Units.largeSpacing

        // --- Preview Card ---
        Kirigami.ShadowedRectangle {
            id: previewCard

            Layout.alignment: Qt.AlignHCenter
            Layout.fillWidth: true
            Layout.maximumWidth: Kirigami.Units.gridUnit * 25
            
            implicitHeight: width * renderer.physicalHeight / Math.max(1, renderer.physicalWidth)

            Kirigami.Theme.colorSet: Kirigami.Theme.View
            Kirigami.Theme.inherit: false

            color: Kirigami.Theme.backgroundColor
            
            border.color: Qt.rgba(Kirigami.Theme.textColor.r, Kirigami.Theme.textColor.g, Kirigami.Theme.textColor.b, 0.15)
            border.width: 1
            
            radius: Kirigami.Units.largeSpacing

            shadow.size: 4
            shadow.yOffset: 2
            shadow.color: Qt.rgba(0, 0, 0, 0.1)

            Item {
                id: previewContainer

                anchors.fill: parent
                anchors.margins: Kirigami.Units.smallSpacing

                clip: true

                function resetPreview() {
                    previewPlayback.reset()
                }

                SlideshowPlayer {
                    id: previewPlayback
                    ready: previewShader.ready
                    loading: previewShader.loading
                    enabled: renderer.displayPlaybackSettings.mode === "Slideshow"
                    imageCount: renderer.displayImageCount
                    generation: renderer.displayGeneration
                    slideInterval: (renderer.displayPlaybackSettings.slideInterval || 10) * 1000
                    transitionDuration: renderer.displayPlaybackSettings.transitionDuration || 1500
                }

                Item {
                    id: offscreenBuffer

                    width: 854
                    height: Math.max(1, Math.round(width * renderer.physicalHeight
                        / Math.max(1, renderer.physicalWidth)))

                    AsciiShader {
                        id: previewShader
                        anchors.fill: parent

                        renderDirectory: renderer.displayDirectory
                        generation: renderer.displayGeneration
                        manifest: renderer.displayManifest
                        imageCount: renderer.displayImageCount
                        currentIndex: previewPlayback.currentIndex
                        nextIndex: previewPlayback.nextIndex
                        transitionProgress: previewPlayback.transitionProgress
                        transitionFrames: renderer.displayFrameCount
                    }
                }

                Rectangle {
                    id: maskRect
                    anchors.fill: parent
                    radius: Kirigami.Units.mediumSpacing
                    
                    visible: false
                    layer.enabled: true
                }

                ShaderEffectSource {
                    id: previewImage
                    sourceItem: offscreenBuffer

                    anchors.fill: parent
                    textureSize: Qt.size(offscreenBuffer.width, offscreenBuffer.height)
                    format: ShaderEffectSource.RGBA8

                    mipmap: false
                    smooth: true
                    samples: 0

                    live: true
                    hideSource: true
                    recursive: false
                    
                    visible: false
                }

                MultiEffect {
                    anchors.fill: parent
                    source: previewImage
                    maskEnabled: true
                    maskSource: maskRect
                    visible: previewShader.ready
                }

                Controls.Label {
                    anchors.centerIn: parent
                    text: "Select an image to render"
                    opacity: 0.5
                    visible: !renderer.displayDirectory && !previewShader.ready
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

        ColumnLayout {
            Layout.alignment: Qt.AlignHCenter
            Layout.fillWidth: true
            Layout.maximumWidth: Kirigami.Units.gridUnit * 25
            spacing: Kirigami.Units.smallSpacing

            Controls.ProgressBar {
                Layout.fillWidth: true
                from: 0
                to: 1
                value: renderer.progress
                indeterminate: renderer.renderState === "connecting" || (renderer.renderState === "running" && renderer.progress === 0)
                visible: renderer.renderState !== "idle"
            }

            RowLayout {
                Layout.fillWidth: true
                Controls.Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: renderer.renderState === "error" ? renderer.error
                        : renderer.renderState === "ready" ? "Render complete"
                        : renderer.stage
                }
                Controls.Button {
                    text: renderer.renderState === "error" ? "Retry" : "Render now"
                    onClicked: renderer.start(renderer.renderState === "ready")
                    enabled: root.editImages.length > 0
                }
            }
        }

        // --- Options ---
        Kirigami.FormLayout {
            Layout.fillWidth: true

            Kirigami.Separator {
                Kirigami.FormData.isSection: true
                Kirigami.FormData.label: "Source"
            }

            Controls.ComboBox {
                id: modeCombo
                Kirigami.FormData.label: "Source Type:"
                Layout.fillWidth: true

                model: ["Image", "Slideshow"]
                currentIndex: model.indexOf(root.editMode)

                onActivated: index => {
                    const newMode = model[index]
                    if (root.editProfile === 0) root.cfg_Mode = newMode
                    else root.cfg_ModeB = newMode

                    if (newMode !== "Slideshow" && imageList.count === 0) {
                        imageList.append({ "path": "" })
                        root.syncToConfig()
                    }

                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            ColumnLayout {
                Kirigami.FormData.isSection: true

                Layout.fillWidth: true
                spacing: Kirigami.Units.smallSpacing

                Controls.ScrollView {
                    id: imageScrollView

                    Layout.fillWidth: true

                    Layout.preferredHeight: Math.min(imageListColumn.implicitHeight, Kirigami.Units.gridUnit * 13)
                    Layout.maximumHeight: Kirigami.Units.gridUnit * 13

                    clip: true
                    Controls.ScrollBar.horizontal.policy: Controls.ScrollBar.AlwaysOff
                    Controls.ScrollBar.vertical.policy: Controls.ScrollBar.AsNeeded

                    ColumnLayout {
                        id: imageListColumn
                        width: imageScrollView.availableWidth
                        spacing: Kirigami.Units.smallSpacing

                        Repeater {
                            model: imageList

                            delegate: RowLayout {
                                id: imageRow
                                required property int index
                                required property string path
                                Layout.fillWidth: true
                                visible: root.editMode === "Slideshow" || imageRow.index === 0

                                ColumnLayout {
                                    spacing: -5

                                    Layout.preferredHeight: Kirigami.Units.gridUnit * 2
                                    Layout.preferredWidth: Kirigami.Units.gridUnit * 1.5

                                    Controls.ToolButton {
                                        icon.name: "arrow-up"
                                        visible: root.editMode === "Slideshow"
                                        enabled: visible && imageRow.index > 0

                                        Layout.fillWidth: true
                                        Layout.fillHeight: true

                                        onClicked: {
                                            imageList.move(imageRow.index, imageRow.index - 1, 1);

                                            root.syncToConfig();
                                            previewContainer.resetPreview();
                                        }
                                    }

                                    Controls.ToolButton {
                                        icon.name: "arrow-down"
                                        visible: root.editMode === "Slideshow"
                                        enabled: visible && imageRow.index < imageList.count - 1

                                        Layout.fillWidth: true
                                        Layout.fillHeight: true

                                        onClicked: {
                                            imageList.move(imageRow.index, imageRow.index + 1, 1);

                                            root.syncToConfig();
                                            previewContainer.resetPreview();
                                        }
                                    }
                                }

                                Controls.TextField {
                                    text: root.cleanPath(imageRow.path)
                                    placeholderText: "Select a file…"
                                    readOnly: true
                                    Layout.fillWidth: true
                                }

                                Controls.Button {
                                    icon.name: "document-open"

                                    visible: root.editMode !== "Slideshow"
                                    enabled: root.editMode !== "Slideshow"

                                    onClicked: {
                                        root._activePicker = imageRow.index;

                                        imagePicker.fileMode = FileDialog.OpenFile;
                                        imagePicker.open();
                                    }
                                }

                                Controls.Button {
                                    icon.name: "list-remove"

                                    enabled: imageList.count > 1
                                    visible: root.editMode === "Slideshow" ? true : false

                                    onClicked: {
                                        root.removeImage(imageRow.index)
                                    }
                                }
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true

                    visible: root.editMode === "Slideshow"

                    Controls.Button {
                        Kirigami.FormData.label: ""

                        Layout.fillWidth: true
                        Layout.preferredWidth: 0

                        enabled: root.editMode === "Slideshow"
                        
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

                        enabled: root.editMode === "Slideshow"
                        
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
                Kirigami.FormData.label: "Profile Mode"
            }

            Controls.ComboBox {
                Kirigami.FormData.label: "Editing Profile:"
                Layout.fillWidth: true
                model: ["Profile A", "Profile B"]
                currentIndex: root.editProfile
                onActivated: index => root.editProfile = index
            }

            Controls.Switch {
                Kirigami.FormData.label: "Time of Day:"
                text: "Switch profiles automatically"
                checked: root.cfg_ScheduleEnabled
                onToggled: {
                    root.cfg_ScheduleEnabled = checked
                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Manual Profile:"
                visible: !root.cfg_ScheduleEnabled

                Controls.RadioButton {
                    text: "Profile A"
                    checked: root.cfg_ManualProfile === "A"
                    onClicked: {
                        root.cfg_ManualProfile = "A"
                        if (root.configDialog) root.configDialog.changed = true
                    }
                }

                Controls.RadioButton {
                    text: "Profile B"
                    checked: root.cfg_ManualProfile === "B"
                    onClicked: {
                        root.cfg_ManualProfile = "B"
                        if (root.configDialog) root.configDialog.changed = true
                    }
                }
            }

            Controls.TextField {
                Kirigami.FormData.label: "Profile A From:"
                visible: root.cfg_ScheduleEnabled
                text: root.cfg_ProfileAStart
                inputMask: "99:99"
                onEditingFinished: {
                    if (/^([01][0-9]|2[0-3]):[0-5][0-9]$/.test(text)) {
                        root.cfg_ProfileAStart = text
                        if (root.configDialog) root.configDialog.changed = true
                    } else text = root.cfg_ProfileAStart
                }
            }

            Controls.TextField {
                Kirigami.FormData.label: "Profile B From:"
                visible: root.cfg_ScheduleEnabled
                text: root.cfg_ProfileBStart
                inputMask: "99:99"
                onEditingFinished: {
                    if (/^([01][0-9]|2[0-3]):[0-5][0-9]$/.test(text)) {
                        root.cfg_ProfileBStart = text
                        if (root.configDialog) root.configDialog.changed = true
                    } else text = root.cfg_ProfileBStart
                }
            }

            Kirigami.Separator {
                Kirigami.FormData.isSection: true
                Kirigami.FormData.label: "Style"
            }

            Controls.ComboBox {
                Kirigami.FormData.label: "Render Style:"
                Layout.fillWidth: true
                model: ["Headroom aware", "Structure first"]
                currentIndex: root.activeSettings.RenderStyle === "structure" ? 1 : 0
                onActivated: index => {
                    const style = index === 1 ? "structure" : "headroom"
                    root.setProfileSetting("RenderStyle", style)
                    root.applyStyleCharacters(style)
                }
            }

            Controls.CheckBox {
                id: invertInkCheck
                Kirigami.FormData.label: "Ink Polarity:"
                text: "Dark ink on a light canvas"
                checked: root.activeSettings.InvertInk
                onToggled: if (root.controlsReady) root.setProfileSetting("InvertInk", checked)
            }

            Controls.CheckBox {
                id: grayscaleCheck
                Kirigami.FormData.label: "Grayscale:"
                checked: root.activeSettings.Grayscale
                onToggled: if (root.controlsReady) root.setProfileSetting("Grayscale", checked)
            }

            Kirigami.Separator {
                Kirigami.FormData.isSection: true
                Kirigami.FormData.label: "Appearance"
            }

            Controls.TextField {
                Kirigami.FormData.label: "Render Cache:"
                Layout.fillWidth: true
                text: root.activeSettings.RenderDirectory
                placeholderText: "Generated automatically after rendering"
                readOnly: true
                onEditingFinished: {
                    root.setProfileSetting("RenderDirectory", text.trim())
                }
            }

            Controls.Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: "Changes render in the background. The last saved ASCII render stays visible until replacement assets are ready."
            }

            Controls.ComboBox {
                id: fontCombo
                Layout.fillWidth: true
                editable: true

                Kirigami.FormData.label: "Font Family:"

                model: Array.from(new Set([root.activeSettings.FontFamily].concat(root.monoFonts)))
                currentIndex: find(root.activeSettings.FontFamily)
                onDownChanged: if (down) root.startFontScan()

                onActivated: {
                    root.setProfileSetting("FontFamily", currentText)
                }

                onAccepted: {
                    root.setProfileSetting("FontFamily", editText)
                }
            }

            ColumnLayout {
                Kirigami.FormData.label: "Characters:"
                Layout.fillWidth: true

                Grid {
                    Layout.fillWidth: true

                    columns: 12
                    spacing: Kirigami.Units.smallSpacing

                    Repeater {
                        model: root.activeSettings.AsciiChars

                        delegate: Controls.TextField {
                            id: glyphEditor
                            required property int index
                            required property string modelData
                            width: Kirigami.Units.gridUnit * 1.15
                            height: Kirigami.Units.gridUnit * 1.45

                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                                                        
                            text: modelData
                            font.family: root.activeSettings.FontFamily

                            maximumLength: 1

                            color: activeFocus ? Kirigami.Theme.highlightedTextColor : Kirigami.Theme.textColor
                            background: Rectangle {
                                radius: Kirigami.Units.smallSpacing
                                color: glyphEditor.activeFocus
                                    ? Kirigami.Theme.highlightColor
                                    : Kirigami.Theme.alternateBackgroundColor
                                border.width: 1
                                border.color: glyphEditor.activeFocus
                                    ? Kirigami.Theme.highlightColor
                                    : Qt.alpha(Kirigami.Theme.textColor, 0.2)
                            }

                            onEditingFinished: {
                                let arr = Array.from(root.activeSettings.AsciiChars)

                                arr[index] = text
                                root.setProfileSetting("AsciiChars", arr)
                            }
                        }
                    }
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Detail Level:"

                Controls.ComboBox {
                    id: detailPreset
                    Layout.fillWidth: true
                    model: ["Ultra high (~320 columns)", "Very high (~240 columns)",
                        "High (~180 columns)", "Mid (~120 columns)", "Low (~80 columns)"]
                    currentIndex: root.detailValues.indexOf(root.activeSettings.Detail)
                    onActivated: index => {
                        root.setProfileSetting("Detail", root.detailValues[index])
                    }
                }

                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: {
                        const columns = [180, 120, 80, 240, 320][root.activeSettings.Detail];
                        const physicalWidth = Math.round(Screen.width * Screen.devicePixelRatio);
                        const physicalHeight = Math.round(Screen.height * Screen.devicePixelRatio);
                        const cellWidth = Math.min(127, Math.max(2, Math.round(physicalWidth / columns)));
                        const rows = Math.ceil(physicalHeight / (cellWidth * 2));
                        const actualColumns = Math.ceil(physicalWidth / cellWidth);
                        return actualColumns + "×" + rows;
                    }
                    font.weight: Font.Bold
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Dithering Strength:"
                visible: root.activeSettings.RenderStyle === "headroom"

                Controls.Slider {
                    id: ditherSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 0.05
                    stepSize: 0.001
                    value: root.activeSettings.Dither
                    onMoved: root.setProfileSetting("Dither", value)
                }
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: ditherSlider.value.toFixed(3)
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Sobel Edge Loss:"
                visible: root.activeSettings.RenderStyle === "headroom"

                Controls.Slider {
                    id: gradientLossSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 4
                    stepSize: 0.1
                    value: root.activeSettings.GradientLossStrength
                    onMoved: root.setProfileSetting("GradientLossStrength", value)
                }
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: gradientLossSlider.value.toFixed(1)
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Glyph Luminosity Bonus:"
                visible: root.activeSettings.RenderStyle === "headroom"

                Controls.Slider {
                    id: glyphLuminanceBonusSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 0.12
                    stepSize: 0.0025
                    value: root.activeSettings.GlyphLuminanceBonus
                    onMoved: root.setProfileSetting("GlyphLuminanceBonus", value)
                }
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: glyphLuminanceBonusSlider.value.toFixed(4)
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Structure Dither:"
                visible: root.activeSettings.RenderStyle === "structure"

                Controls.Slider {
                    id: structureDitherSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 0.12
                    stepSize: 0.005
                    value: root.activeSettings.StructureDither
                    onMoved: root.setProfileSetting("StructureDither", value)
                }
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: structureDitherSlider.value.toFixed(3)
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Background:"

                Controls.ComboBox {
                    Layout.fillWidth: true
                    model: ["Image darks (uniform)", "Choose a color"]
                    currentIndex: root.activeSettings.BackgroundMode
                    onActivated: index => {
                        root.setProfileSetting("BackgroundMode", index)
                    }
                }
                Rectangle {
                    visible: root.activeSettings.BackgroundMode === 1
                    Layout.preferredWidth: 24
                    Layout.preferredHeight: 24
                    radius: 3
                    color: root.activeSettings.BackgroundColor
                    border.color: Kirigami.Theme.textColor
                }
                Controls.Button {
                    visible: root.activeSettings.BackgroundMode === 1
                    text: root.activeSettings.BackgroundColor
                    onClicked: backgroundPicker.open()
                }
            }

            Kirigami.Separator {
                Kirigami.FormData.isSection: true
                Kirigami.FormData.label: "Transition"
                visible: root.editMode === "Slideshow"
            }

            Controls.SpinBox {
                Kirigami.FormData.label: "Transition Frames:"
                visible: root.editMode === "Slideshow"
                from: 2
                to: 120
                value: root.activeSettings.TransitionFrames
                onValueModified: root.setProfileSetting("TransitionFrames", value)
            }

            Controls.SpinBox {
                Kirigami.FormData.label: "Transition Time (ms):"
                visible: root.editMode === "Slideshow"
                from: 100
                to: 30000
                stepSize: 100
                value: root.activeSettings.TransitionDuration
                onValueModified: root.setProfileSetting("TransitionDuration", value)
            }

            Controls.ComboBox {
                Kirigami.FormData.label: "Transition Pacing:"
                visible: root.editMode === "Slideshow"
                Layout.fillWidth: true
                model: ["Linear", "Ease in", "Ease out", "Ease in and out"]
                currentIndex: ["linear", "ease-in", "ease-out", "ease-in-out"].indexOf(root.activeSettings.TransitionPace)
                onActivated: index => root.setProfileSetting("TransitionPace", ["linear", "ease-in", "ease-out", "ease-in-out"][index])
            }

            Controls.ComboBox {
                Kirigami.FormData.label: "Morph Pattern:"
                visible: root.editMode === "Slideshow"
                Layout.fillWidth: true
                model: ["Blue noise dissolve", "Ripple", "Spiral", "Directional flow", "From edges", "Image salience"]
                currentIndex: ["blue-noise", "ripple", "spiral", "directional", "edge", "salience"].indexOf(root.activeSettings.TransitionField)
                onActivated: index => root.setProfileSetting("TransitionField", ["blue-noise", "ripple", "spiral", "directional", "edge", "salience"][index])
            }

            RowLayout {
                Kirigami.FormData.label: "Spatial Stagger:"
                visible: root.editMode === "Slideshow"
                Controls.Slider {
                    id: staggerSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 0.9
                    stepSize: 0.05
                    value: root.activeSettings.TransitionSpread
                    onMoved: root.setProfileSetting("TransitionSpread", value)
                }
                Controls.Label { text: staggerSlider.value.toFixed(2) }
            }

            RowLayout {
                Kirigami.FormData.label: "Slide Interval:"
                visible: root.editMode === "Slideshow"

                Controls.Slider {
                    id: intervalSlider
                    Layout.fillWidth: true
                    snapMode: Controls.Slider.SnapOnRelease
                    stepSize: 2.5
                    from: 5
                    to: 30
                    value: root.activeSettings.SlideInterval
                    onValueChanged: previewContainer.resetPreview()
                    onMoved: root.setProfileSetting("SlideInterval", value)
                }
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 7
                    text: intervalSlider.value + " seconds"
                    font.weight: Font.Bold
                }
            }
        }
    }
}
