/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick

Item {
    id: client
    property string serviceUrl: "http://127.0.0.1:48741"
    property int maximum: 20
    property bool autoSync: false
    property bool busy: false
    property string message: ""
    property var activeRequest: null
    onMaximumChanged: if (autoSync) sync.restart()
    Component.onCompleted: if (autoSync) sync.restart()
    Component.onDestruction: if (activeRequest) activeRequest.abort()

    Timer { id: sync; interval: 0; onTriggered: client.setMaximum() }
    Timer {
        id: timeout
        interval: 10000
        onTriggered: {
            const previous = client.activeRequest
            client.activeRequest = null
            client.busy = false
            if (previous) previous.abort()
            client.message = "Cache request timed out."
        }
    }

    function setMaximum() { send("/v1/cache", {maximum: maximum}, false) }
    function clear() { send("/v1/cache/clear", {}, true) }

    function send(path, payload, clearing) {
        if (activeRequest) {
            const previous = activeRequest
            activeRequest = null
            previous.abort()
        }
        const xhr = new XMLHttpRequest()
        activeRequest = xhr
        busy = true
        message = ""
        xhr.open("POST", serviceUrl + path, true)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || client.activeRequest !== xhr) return
            client.activeRequest = null
            client.busy = false
            timeout.stop()
            try {
                const reply = JSON.parse(xhr.responseText)
                if (xhr.status !== 200) client.message = reply.error || "Cache request failed."
                else if (clearing) client.message = "Cache cleared. " + reply.count + " in-use render(s) retained."
            } catch (failure) { client.message = "Renderer service is unavailable." }
        }
        timeout.restart()
        xhr.send(JSON.stringify(payload))
    }
}
