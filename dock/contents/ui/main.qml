/* SPDX-License-Identifier: AGPL-3.0-or-later */

pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Layouts
import org.kde.plasma.plasmoid

PlasmoidItem {
    id: root
    property string profile: "auto"
    property string error: ""
    property var activeRequest: null
    preferredRepresentation: compactRepresentation
    toolTipMainText: "ASCII Wallpaper Profile"
    toolTipSubText: profile === "auto" ? "Scheduled" : "Profile " + profile + " (manual override)"

    function request(method, next) {
        if (activeRequest) {
            if (method === "GET") return
            const previous = activeRequest
            activeRequest = null
            previous.abort()
        }
        const xhr = new XMLHttpRequest()
        activeRequest = xhr
        xhr.open(method, "http://127.0.0.1:48741/v1/override", true)
        if (method === "POST") xhr.setRequestHeader("Content-Type", "application/json")
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || root.activeRequest !== xhr) return
            root.activeRequest = null
            requestTimeout.stop()
            if (xhr.status === 200) {
                try {
                    root.profile = JSON.parse(xhr.responseText).profile
                    root.error = ""
                } catch (failure) { root.error = String(failure) }
            } else root.error = "Renderer service is unavailable"
        }
        requestTimeout.restart()
        xhr.send(method === "POST" ? JSON.stringify({profile: next}) : null)
    }

    function toggleProfile(target) {
        root.request("POST", root.profile === target ? "auto" : target)
    }

    Timer {
        id: requestTimeout
        interval: 5000
        onTriggered: {
            const previous = root.activeRequest
            root.activeRequest = null
            if (previous) previous.abort()
            root.error = "Renderer request timed out"
        }
    }

    Timer {
        interval: 2000
        repeat: true
        running: true
        onTriggered: root.request("GET", "")
    }
    Component.onCompleted: request("GET", "")
    Component.onDestruction: {
        const previous = activeRequest
        activeRequest = null
        if (previous) previous.abort()
    }

    compactRepresentation: Controls.Button {
        text: root.profile === "auto" ? "A/B" : root.profile
        icon.name: "preferences-desktop-wallpaper"
        onClicked: root.request("POST", root.profile === "A" ? "B" : "A")
        onPressAndHold: root.expanded = true
        Controls.ToolTip.visible: hovered
        Controls.ToolTip.text: root.profile === "auto" ? "Use profile A; open for schedule controls"
            : "Switch to profile " + (root.profile === "A" ? "B" : "A")
    }

    fullRepresentation: ColumnLayout {
        spacing: 8
        Controls.Label { text: "ASCII Wallpaper Profile"; font.bold: true }
        Controls.Label { text: root.error || (root.profile === "auto" ? "Following time of day" : "Manual override: " + root.profile) }
        RowLayout {
            Controls.Button { text: "A"; checkable: true; checked: root.profile === "A"; onClicked: root.toggleProfile("A") }
            Controls.Button { text: "B"; checkable: true; checked: root.profile === "B"; onClicked: root.toggleProfile("B") }
        }
        Controls.Button { text: "Follow schedule"; enabled: root.profile !== "auto"; onClicked: root.request("POST", "auto") }
    }
}
