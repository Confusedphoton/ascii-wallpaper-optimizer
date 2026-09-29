/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtCore
import QtQuick
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
    readonly property string standardChars: " .´:rszunvcxLOGDQ0KNMW&@\\=^"
    readonly property string legacyStandardChars: " .´:rszunvcxLOGDQ0KNMW&@"
    readonly property string structureChars: " .´:-_/\\=^|+rszunvcxLOGDQ0KNMW&@[]{}"
    readonly property string legacyStructureChars: " .´:-_/|+rszunvcxLOGDQ0KNMW&@[]{}"
    property list<string> cfg_AsciiChars: [" ", ".", "´", ":", "r", "s", "z", "u", "n", "v", "c", "x", "L", "O", "G", "D", "Q", "0", "K", "N", "M", "W", "&", "@", "\\", "=", "^"]
    property list<string> cfg_Images: []
    property var monoFonts: [cfg_FontFamily]
    property var fontFamiliesToCheck: []
    property var foundMonoFonts: []
    property int fontScanIndex: 0
    property bool fontScanStarted: false
    property int renderRevision: 0
    property bool controlsReady: false

    function colorHex(color) {
        return "#" + [color.r, color.g, color.b].map(channel =>
            Math.round(channel * 255).toString(16).padStart(2, "0")).join("")
    }

    function applyStyleCharacters(style) {
        const current = Array.from(cfg_AsciiChars).join("")
        if (style === "structure" && (current === standardChars || current === legacyStandardChars
                || current === legacyStructureChars))
            cfg_AsciiChars = Array.from(structureChars)
        else if (style === "headroom" && (current === legacyStandardChars
                || current === structureChars || current === legacyStructureChars))
            cfg_AsciiChars = Array.from(standardChars)
    }

    property int cfg_Detail: 1
    readonly property var detailValues: [4, 3, 0, 1, 2]
    property alias cfg_Grayscale: grayscaleCheck.checked
    property alias cfg_SlideInterval: intervalSlider.value
    property alias cfg_Dither: ditherSlider.value
    property alias cfg_GradientLossStrength: gradientLossSlider.value
    property alias cfg_StructureDither: structureDitherSlider.value
    property alias cfg_InvertInk: invertInkCheck.checked

    property string cfg_Image: ""

    ListModel { id: imageList }
    property int _activePicker: -1
    property bool _isInternalUpdate: false

    onCfg_ImagesChanged: {
        if (editProfile === 0) loadFromConfig();
    }
    onCfg_ImagesBChanged: if (editProfile === 1) loadFromConfig()
    onEditProfileChanged: {
        loadFromConfig()
        if (controlsReady) previewContainer.resetPreview()
    }
    onCfg_FontFamilyChanged: {
        if (!monoFonts.includes(cfg_FontFamily))
            monoFonts = [cfg_FontFamily].concat(monoFonts)
        fontCombo.editText = cfg_FontFamily
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

        let currentChars = Array.from(root.cfg_AsciiChars || []);
        if (currentChars.length && currentChars[0] === "\\s") {
            currentChars[0] = " ";
            migrated = true;
            root.cfg_AsciiChars = currentChars;
        }
        if (currentChars.length === 23 && currentChars[0] !== " ") {
            currentChars.unshift(" ");

            migrated = true;
            root.cfg_AsciiChars = currentChars;
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
        loadFromConfig()
        applyStyleCharacters(cfg_RenderStyle)
        fontCombo.editText = cfg_FontFamily
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
                root.monoFonts = Array.from(new Set([root.cfg_FontFamily].concat(root.foundMonoFonts))).sort()
            }
        }
    }

    RenderClient {
        id: renderer
        images: root.editMode === "Slideshow" ? Array.from(root.editImages) : Array.from(root.editImages).slice(0, 1)
        fontFamily: root.cfg_FontFamily
        characters: Array.from(root.cfg_AsciiChars).join("")
        detail: root.cfg_Detail
        grayscale: root.cfg_Grayscale
        frames: root.cfg_TransitionFrames
        transitionPace: root.cfg_TransitionPace
        transitionField: root.cfg_TransitionField
        transitionSpread: root.cfg_TransitionSpread
        dither: root.cfg_Dither
        gradientLossStrength: root.cfg_RenderStyle === "headroom" ? root.cfg_GradientLossStrength : 0
        structureDither: root.cfg_StructureDither
        renderStyle: root.cfg_RenderStyle
        invertInk: root.cfg_InvertInk
        background: root.cfg_BackgroundMode === 1 ? root.cfg_BackgroundColor : "auto"
        cachedDirectory: root.cfg_RenderDirectory
        onRenderStateChanged: {
            if (renderState === "ready") {
                root.renderRevision++
                if (root.cfg_RenderDirectory !== directory) {
                    root.cfg_RenderDirectory = directory
                    if (root.configDialog) root.configDialog.changed = true
                }
            }
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
                        imageList.append({ "path": clean });
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
                        imageList.setProperty(_activePicker, "path", clean)
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

    ColorDialog {
        id: backgroundPicker
        title: "Select background color"
        selectedColor: root.cfg_BackgroundColor
        onAccepted: {
            root.cfg_BackgroundColor = root.colorHex(selectedColor)
            if (root.configDialog) root.configDialog.changed = true
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

                property int currentIdx: 0
                property int nextIdx: 0

                function resetPreview() {
                    previewTimer.stop();
                    crossfadeAnimation.stop();
                    previewShader.transitionProgress = 0.0;
                    currentIdx = 0;
                    nextIdx = 0;
                    previewTimer.restart();
                }

                onCurrentIdxChanged: previewShader.transitionProgress = 0.0
                Connections {
                    target: root
                    function onCfg_ModeChanged() { previewContainer.resetPreview() }
                    function onCfg_ModeBChanged() { previewContainer.resetPreview() }
                }

                function getPreviewPath(index) {
                    if (imageList.count === 0 || index < 0 || index >= imageList.count) {
                        return "";
                    }

                    let path = imageList.get(index).path;
                    return root.toFileUrl(path);
                }

                Timer {
                    id: previewTimer
                    
                    interval: root.cfg_SlideInterval * 1000
                    running: root.editMode === "Slideshow" && imageList.count > 1
                        && previewContainer.currentIdx === previewContainer.nextIdx
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

                    duration: root.cfg_TransitionDuration
                    easing.type: Easing.Linear
                    
                    onFinished: {
                        previewContainer.currentIdx = previewContainer.nextIdx;
                        previewShader.transitionProgress = 0.0;
                    }
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
                        currentIndex: previewContainer.currentIdx
                        nextIndex: previewContainer.nextIdx
                        transitionFrames: root.cfg_TransitionFrames
                        detail: root.cfg_Detail
                        revision: root.renderRevision
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
                    visible: renderer.renderState === "ready" && previewShader.ready
                }

                Controls.Label {
                    anchors.centerIn: parent
                    text: "Select an image to render"
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

            RowLayout {
                Kirigami.FormData.isSection: true
                Layout.fillWidth: true
                spacing: Kirigami.Units.mediumSpacing

                Controls.Label {
                    text: "Mode:"
                }

                Controls.ComboBox {
                    id: modeCombo
                    Layout.fillWidth: true

                    model: ["Image", "Slideshow"]
                    currentIndex: model.indexOf(root.editMode)

                    onActivated: (index) => {
                        let newMode = model[index];
                        if (root.editProfile === 0) root.cfg_Mode = newMode;
                        else root.cfg_ModeB = newMode;

                        if (newMode !== "Slideshow" && imageList.count === 0) {
                            imageList.append({ "path": "" });
                            root.syncToConfig();
                        }

                        if (root.configDialog) {
                            root.configDialog.changed = true;
                        }
                    }
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
                                Layout.fillWidth: true
                                visible: root.editMode === "Slideshow" || index === 0

                                ColumnLayout {
                                    spacing: -5

                                    Layout.preferredHeight: Kirigami.Units.gridUnit * 2
                                    Layout.preferredWidth: Kirigami.Units.gridUnit * 1.5

                                    Controls.ToolButton {
                                        icon.name: "arrow-up"
                                        visible: root.editMode === "Slideshow"
                                        enabled: visible && index > 0

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
                                        visible: root.editMode === "Slideshow"
                                        enabled: visible && index < imageList.count - 1

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

                                    visible: root.editMode !== "Slideshow"
                                    enabled: root.editMode !== "Slideshow"

                                    onClicked: {
                                        root._activePicker = index;

                                        imagePicker.fileMode = FileDialog.OpenFile;
                                        imagePicker.open();
                                    }
                                }

                                Controls.Button {
                                    icon.name: "list-remove"

                                    enabled: imageList.count > 1
                                    visible: root.editMode === "Slideshow" ? true : false

                                    onClicked: {
                                        root.removeImage(index)
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
                Kirigami.FormData.label: "Appearance"
            }

            Controls.TextField {
                Kirigami.FormData.label: "Render Cache:"
                Layout.fillWidth: true
                text: root.cfg_RenderDirectory
                placeholderText: "Generated automatically after rendering"
                readOnly: true
                onEditingFinished: {
                    root.cfg_RenderDirectory = text.trim()
                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            Controls.Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: "Changes render in the background. The last saved ASCII render stays visible until replacement assets are ready."
            }

            Controls.ComboBox {
                Kirigami.FormData.label: "Render Style:"
                Layout.fillWidth: true
                model: ["Headroom aware", "Structure first"]
                currentIndex: root.cfg_RenderStyle === "structure" ? 1 : 0
                onActivated: index => {
                    root.cfg_RenderStyle = index === 1 ? "structure" : "headroom"
                    root.applyStyleCharacters(root.cfg_RenderStyle)
                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            Controls.CheckBox {
                id: invertInkCheck
                Kirigami.FormData.label: "Ink Polarity:"
                text: "Dark ink on a light canvas"
            }

            Controls.SpinBox {
                Kirigami.FormData.label: "Transition Frames:"
                from: 2
                to: 120
                value: root.cfg_TransitionFrames
                onValueModified: {
                    root.cfg_TransitionFrames = value
                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            Controls.SpinBox {
                Kirigami.FormData.label: "Transition Time (ms):"
                from: 100
                to: 30000
                stepSize: 100
                value: root.cfg_TransitionDuration
                onValueModified: {
                    root.cfg_TransitionDuration = value
                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            Controls.ComboBox {
                Kirigami.FormData.label: "Transition Pacing:"
                Layout.fillWidth: true
                model: ["Linear", "Ease in", "Ease out", "Ease in and out"]
                currentIndex: ["linear", "ease-in", "ease-out", "ease-in-out"].indexOf(root.cfg_TransitionPace)
                onActivated: index => {
                    root.cfg_TransitionPace = ["linear", "ease-in", "ease-out", "ease-in-out"][index]
                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            Controls.ComboBox {
                Kirigami.FormData.label: "Morph Pattern:"
                Layout.fillWidth: true
                model: ["Blue noise dissolve", "Ripple", "Spiral", "Directional flow", "From edges", "Image salience"]
                currentIndex: ["blue-noise", "ripple", "spiral", "directional", "edge", "salience"].indexOf(root.cfg_TransitionField)
                onActivated: index => {
                    root.cfg_TransitionField = ["blue-noise", "ripple", "spiral", "directional", "edge", "salience"][index]
                    if (root.configDialog) root.configDialog.changed = true
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Spatial Stagger:"
                Controls.Slider {
                    id: staggerSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 0.9
                    stepSize: 0.05
                    value: root.cfg_TransitionSpread
                    onMoved: {
                        root.cfg_TransitionSpread = value
                        if (root.configDialog) root.configDialog.changed = true
                    }
                }
                Controls.Label { text: staggerSlider.value.toFixed(2) }
            }

            Controls.ComboBox {
                id: fontCombo
                Layout.fillWidth: true
                editable: true

                Kirigami.FormData.label: "Font Family:"

                model: Array.from(new Set([root.cfg_FontFamily].concat(root.monoFonts)))
                currentIndex: find(root.cfg_FontFamily)
                onDownChanged: if (down) root.startFontScan()

                onActivated: {
                    root.cfg_FontFamily = currentText
                    if (root.configDialog) root.configDialog.changed = true
                }

                onAccepted: {
                    root.cfg_FontFamily = editText
                    if (root.configDialog) root.configDialog.changed = true
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
                        model: root.cfg_AsciiChars

                        delegate: Controls.TextField {
                            width: Kirigami.Units.gridUnit * 1.15
                            height: Kirigami.Units.gridUnit * 1.45

                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                                                        
                            text: modelData
                            font.family: root.cfg_FontFamily

                            maximumLength: 1

                            color: activeFocus ? Kirigami.Theme.highlightedTextColor : Kirigami.Theme.textColor
                            background: Rectangle {
                                radius: Kirigami.Units.smallSpacing
                                color: activeFocus
                                    ? Kirigami.Theme.highlightColor
                                    : Kirigami.Theme.alternateBackgroundColor
                                border.width: 1
                                border.color: activeFocus
                                    ? Kirigami.Theme.highlightColor
                                    : Qt.alpha(Kirigami.Theme.textColor, 0.2)
                            }

                            onEditingFinished: {
                                let arr = Array.from(root.cfg_AsciiChars)

                                arr[index] = text
                                root.cfg_AsciiChars = arr

                                if (root.configDialog) {
                                    root.configDialog.changed = true
                                }
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
                    currentIndex: root.detailValues.indexOf(root.cfg_Detail)
                    onActivated: index => {
                        root.cfg_Detail = root.detailValues[index]
                        if (root.configDialog) root.configDialog.changed = true
                    }
                }

                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: {
                        const columns = [180, 120, 80, 240, 320][root.cfg_Detail];
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
                visible: root.cfg_RenderStyle === "headroom"

                Controls.Slider {
                    id: ditherSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 0.05
                    stepSize: 0.001
                    onValueChanged: {
                        if (root.controlsReady && root.configDialog)
                            root.configDialog.changed = true
                    }
                }
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: ditherSlider.value.toFixed(3)
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Sobel Edge Loss:"
                visible: root.cfg_RenderStyle === "headroom"

                Controls.Slider {
                    id: gradientLossSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 4
                    stepSize: 0.1
                    value: 1.5
                    onValueChanged: {
                        if (root.controlsReady && root.configDialog)
                            root.configDialog.changed = true
                    }
                }
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 10
                    text: gradientLossSlider.value.toFixed(1)
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Structure Dither:"
                visible: root.cfg_RenderStyle === "structure"

                Controls.Slider {
                    id: structureDitherSlider
                    Layout.fillWidth: true
                    from: 0
                    to: 0.12
                    stepSize: 0.005
                    onValueChanged: {
                        if (root.controlsReady && root.configDialog)
                            root.configDialog.changed = true
                    }
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
                    currentIndex: root.cfg_BackgroundMode
                    onActivated: index => {
                        root.cfg_BackgroundMode = index
                        if (root.configDialog) root.configDialog.changed = true
                    }
                }
                Rectangle {
                    visible: root.cfg_BackgroundMode === 1
                    Layout.preferredWidth: 24
                    Layout.preferredHeight: 24
                    radius: 3
                    color: root.cfg_BackgroundColor
                    border.color: Kirigami.Theme.textColor
                }
                Controls.Button {
                    visible: root.cfg_BackgroundMode === 1
                    text: root.cfg_BackgroundColor
                    onClicked: backgroundPicker.open()
                }
            }

            RowLayout {
                Kirigami.FormData.label: "Interval:"
                visible: root.editMode === "Slideshow"

                Controls.Slider {
                    id: intervalSlider
                    Layout.fillWidth: true

                    snapMode: Controls.Slider.SnapOnRelease
                    stepSize: 2.5

                    from: 5
                    to: 30

                    onValueChanged: {
                        previewContainer.resetPreview()
                    }
                }
                
                Controls.Label {
                    Layout.preferredWidth: Kirigami.Units.gridUnit * 7

                    text: intervalSlider.value + " seconds"
                    font.weight: Font.Bold
                }
            }

            Controls.CheckBox {
                id: grayscaleCheck
                Kirigami.FormData.label: "Grayscaled:"
            }
        }
    }
}
