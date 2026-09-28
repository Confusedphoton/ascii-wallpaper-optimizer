/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick
import QtQuick.Controls as Controls
import QtQuick.Layouts
import org.kde.plasma.plasmoid

PlasmoidItem {
    id: root
    property string profile: "auto"
    property string error: ""
    preferredRepresentation: compactRepresentation
    toolTipMainText: "ASCII Wallpaper Profile"
    toolTipSubText: profile === "auto" ? "Scheduled" : "Profile " + profile + " (manual override)"

    function request(method, next) {
        const xhr = new XMLHttpRequest()
        xhr.open(method, "http://127.0.0.1:48741/v1/override", true)
        if (method === "POST") xhr.setRequestHeader("Content-Type", "application/json")
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE) return
            if (xhr.status === 200) {
                try {
                    root.profile = JSON.parse(xhr.responseText).profile
                    root.error = ""
                } catch (failure) { root.error = String(failure) }
            } else root.error = "Renderer service is unavailable"
        }
        xhr.send(method === "POST" ? JSON.stringify({profile: next}) : null)
    }

    Timer {
        interval: 2000
        repeat: true
        running: true
        onTriggered: root.request("GET", "")
    }
    Component.onCompleted: request("GET", "")

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
            Controls.Button { text: "A"; checkable: true; checked: root.profile === "A"; onClicked: root.request("POST", "A") }
            Controls.Button { text: "B"; checkable: true; checked: root.profile === "B"; onClicked: root.request("POST", "B") }
        }
        Controls.Button { text: "Follow schedule"; enabled: root.profile !== "auto"; onClicked: root.request("POST", "auto") }
    }
}
