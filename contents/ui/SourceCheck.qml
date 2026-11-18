/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick

Item {
    id: check
    property string serviceUrl: ""
    property string inputFormat: "images"
    property var paths: []
    property string status: "empty"
    property string message: ""
    property var info: ({})
    property var request: null
    readonly property string sourceKey: JSON.stringify([inputFormat, paths])

    onSourceKeyChanged: schedule()
    onEnabledChanged: schedule()
    onServiceUrlChanged: schedule()
    Component.onDestruction: abort()

    Timer { id: refresh; interval: 120; onTriggered: check.inspect() }

    Timer {
        id: timeout
        interval: 60000
        onTriggered: {
            check.abort()
            check.message = "Checking the source timed out. Try again."
            check.status = "error"
        }
    }

    function schedule() {
        abort()
        status = enabled && paths.length ? "checking" : "empty"
        refresh.restart()
    }

    function abort() {
        const previous = request
        request = null
        timeout.stop()
        if (previous) previous.abort()
    }

    function inspect() {
        abort()
        info = ({})
        message = ""
        if (!enabled || !serviceUrl || !paths.length) {
            status = "empty"
            return
        }
        status = "checking"
        const key = sourceKey
        const xhr = new XMLHttpRequest()
        request = xhr
        xhr.open("POST", serviceUrl + "/v1/source-info", true)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || check.request !== xhr || key !== check.sourceKey) return
            check.request = null
            timeout.stop()
            try {
                const reply = JSON.parse(xhr.responseText)
                if (xhr.status === 200) {
                    check.info = reply
                    check.status = "valid"
                    return
                }
                check.message = reply.error
            } catch (failure) {}
            check.message = check.message || "Could not check the source. Check that the render service is running."
            check.status = "error"
        }
        timeout.restart()
        xhr.send(JSON.stringify({input_format: inputFormat, images: paths}))
    }
}
