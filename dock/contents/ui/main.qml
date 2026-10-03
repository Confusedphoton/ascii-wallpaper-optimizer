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
    property var profiles: [{id: "A", name: "Profile A"}, {id: "B", name: "Profile B"}]
    readonly property string profileName: (profiles.find(choice => choice.id === profile) || {name: profile}).name
    property var activeRequest: null
    preferredRepresentation: compactRepresentation
    toolTipMainText: "ASCII Wallpaper Profile"
    toolTipSubText: profile === "auto" ? "Scheduled" : profileName + " (manual override)"

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
                    const reply = JSON.parse(xhr.responseText)
                    root.profile = reply.profile
                    if (Array.isArray(reply.profiles) && reply.profiles.length) root.profiles = reply.profiles
                    root.error = ""
                } catch (failure) { root.error = String(failure) }
            } else root.error = "Renderer service is unavailable"
        }
        requestTimeout.restart()
        xhr.send(method === "POST" ? JSON.stringify({profile: next}) : null)
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
        text: root.profile === "auto" ? "Auto" : root.profileName
        icon.name: "preferences-desktop-wallpaper"
        onClicked: root.expanded = !root.expanded
        Controls.ToolTip.visible: hovered
        Controls.ToolTip.text: "Choose a wallpaper profile"
    }

    fullRepresentation: ColumnLayout {
        spacing: 8
        Controls.Label { text: "ASCII Wallpaper Profile"; font.bold: true }
        Controls.Label { text: root.error || (root.profile === "auto" ? "Following time of day" : "Manual override: " + root.profileName) }
        Controls.ComboBox {
            Layout.fillWidth: true
            model: ["Follow schedule"].concat(root.profiles.map(choice => choice.name))
            currentIndex: root.profile === "auto" ? 0 : root.profiles.findIndex(choice => choice.id === root.profile) + 1
            onActivated: index => root.request("POST", index === 0 ? "auto" : root.profiles[index - 1].id)
        }
    }
}
