/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick

Item {
    id: client

    property var images: []
    property string fontFamily: "Monospace"
    property string characters: " .´:rszunvcxLOGDQ0KNMW&@"
    property int detail: 1
    property bool grayscale: false
    property int frames: 16
    property string transitionPace: "ease-in-out"
    property string transitionField: "blue-noise"
    property real transitionSpread: 0.65
    property real dither: 0.005
    property real structureDither: 0.06
    property string renderStyle: "headroom"
    property bool invertInk: false
    property string background: "auto"
    property string screenName: Screen.name || ""
    property int physicalWidth: Math.round(Screen.width * Screen.devicePixelRatio)
    property int physicalHeight: Math.round(Screen.height * Screen.devicePixelRatio)
    property bool autoStart: true

    property string renderState: "idle"
    property string stage: ""
    property string error: ""
    property real progress: 0
    property string directory: ""
    property string jobId: ""
    property int requestSerial: 0

    readonly property string settingsKey: JSON.stringify({
        images: Array.from(images || []), font: fontFamily, chars: characters,
        detail: detail, grayscale: grayscale, frames: frames, dither: dither,
        transition_pace: transitionPace, transition_field: transitionField,
        transition_spread: transitionSpread,
        structure_dither: structureDither, style: renderStyle, invert_ink: invertInk,
        background: background,
        screen: screenName, size: [physicalWidth, physicalHeight]
    })

    onSettingsKeyChanged: {
        requestSerial++
        jobId = ""
        directory = ""
        renderState = "idle"
        if (autoStart) schedule.restart()
    }
    onAutoStartChanged: {
        if (autoStart) schedule.restart()
    }
    Component.onCompleted: {
        if (autoStart) schedule.restart()
    }

    Timer {
        id: schedule
        interval: 800
        repeat: false
        onTriggered: client.start(false)
    }

    Timer {
        id: poll
        interval: 500
        repeat: true
        running: client.renderState === "queued" || client.renderState === "running"
        onTriggered: client.checkStatus()
    }

    function consume(job) {
        stage = job.stage || ""
        error = job.error || ""
        progress = Math.max(0, Math.min(1, Number(job.progress) || 0))
        directory = job.directory || ""
        jobId = job.id || ""
        renderState = job.state || "error"
    }

    function start(force) {
        schedule.stop()
        const request = JSON.parse(settingsKey)
        request.images = request.images.map(path => String(path).trim()).filter(path => path.length)
        if (!request.images.length) {
            renderState = "idle"
            stage = "Select an image"
            directory = ""
            jobId = ""
            return
        }
        request.force = !!force
        const serial = ++requestSerial
        renderState = "connecting"
        stage = "Connecting to renderer"
        error = ""
        const xhr = new XMLHttpRequest()
        xhr.open("POST", "http://127.0.0.1:48741/v1/render", true)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || serial !== client.requestSerial) return
            try {
                const reply = JSON.parse(xhr.responseText)
                if (xhr.status !== 200) throw new Error(reply.error || "Render request failed")
                client.consume(reply)
            } catch (failure) {
                client.renderState = "error"
                client.error = xhr.status === 0
                    ? "Renderer service is unavailable. Run the installer, then retry."
                    : String(failure)
            }
        }
        xhr.send(JSON.stringify(request))
    }

    function checkStatus() {
        if (!jobId) return
        const expected = jobId
        const xhr = new XMLHttpRequest()
        xhr.open("GET", "http://127.0.0.1:48741/v1/status/" + expected, true)
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || expected !== client.jobId) return
            if (xhr.status === 200) {
                try { client.consume(JSON.parse(xhr.responseText)) }
                catch (failure) { client.renderState = "error"; client.error = String(failure) }
            } else {
                client.renderState = "error"
                client.error = "Lost contact with renderer service."
            }
        }
        xhr.send()
    }
}
