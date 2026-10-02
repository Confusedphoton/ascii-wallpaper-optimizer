/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick

Item {
    id: client

    property var requestSettings: ({})
    property var playbackSettings: ({})
    property bool autoRender: false
    property bool initialized: false
    property string cachedDirectory: ""
    property string cachedState: ""
    property string screenName: Screen.name || ""
    property int physicalWidth: Math.round(Screen.width * Screen.devicePixelRatio)
    property int physicalHeight: Math.round(Screen.height * Screen.devicePixelRatio)
    property string serviceUrl: "http://127.0.0.1:48741"
    property string clientId: "qml-" + Date.now() + "-" + Math.random().toString(16).slice(2)
    property int requestTimeout: 10000
    property int retryBaseInterval: 1000
    property int retryMaxInterval: 30000
    property int statusInterval: 500
    property int readyStatusInterval: 3000

    property string renderState: "idle"
    property string stage: ""
    property string error: ""
    property real progress: 0
    property string directory: ""
    property string displayDirectory: ""
    property string displayGeneration: ""
    property int displayImageCount: 0
    property int displayFrameCount: 16
    property var displaySettings: ({})
    property var displayRequestSettings: ({})
    property var displayPlaybackSettings: ({})
    property var displayManifest: ({})
    property string jobId: ""
    property int requestSerial: 0
    property bool currentRequestForced: false
    property bool submissionPending: false
    property bool settingsPending: false
    property bool requestInFlight: false
    property int retryAttempt: 0
    property var pendingSettings: ({})
    property var activeRequest: null
    readonly property bool retrying: retry.running

    readonly property string publishedState: JSON.stringify({directory: displayDirectory,
        generation: displayGeneration, manifest: displayManifest, settings: displaySettings,
        request: displayRequestSettings,
        image_count: displayImageCount, frame_count: displayFrameCount,
        playback: displayPlaybackSettings})

    signal assetsReady()

    readonly property string settingsKey: JSON.stringify(Object.assign({}, requestSettings, {
        screen: screenName, size: [physicalWidth, physicalHeight]
    }))

    onSettingsKeyChanged: {
        cancelRequest()
        currentRequestForced = false
        submissionPending = false
        settingsPending = !displayDirectory || !displayMatchesRequest()
        pendingSettings = ({})
        retryAttempt = 0
        retry.stop()
        jobId = ""
        directory = ""
        renderState = displayDirectory ? "ready" : "idle"
        stage = "Settings changed. Apply to render."
        resumeSavedStatus()
        if (initialized && autoRender) automaticRender.restart()
    }
    onCachedDirectoryChanged: Qt.callLater(restoreCachedAssets)
    onCachedStateChanged: Qt.callLater(restoreCachedAssets)
    onPlaybackSettingsChanged: {
        if (renderState === "ready" && !settingsPending) {
            displayPlaybackSettings = Object.assign({}, playbackSettings)
            Qt.callLater(savePlaybackState)
        }
    }
    Component.onCompleted: {
        restoreCachedAssets()
        initialized = true
        if (autoRender) automaticRender.restart()
    }
    Component.onDestruction: cancelRequest()

    Timer {
        id: automaticRender
        interval: 0
        onTriggered: {
            client.restoreCachedAssets()
            if (!client.displayDirectory || !client.displayMatchesRequest()) client.start(false)
        }
    }

    Timer {
        id: poll
        interval: client.renderState === "ready" ? client.readyStatusInterval : client.statusInterval
        repeat: true
        running: !!client.jobId && (client.renderState === "queued" || client.renderState === "running"
            || client.renderState === "ready")
        onTriggered: client.checkStatus()
    }

    Timer {
        id: retry
        interval: Math.min(client.retryMaxInterval,
            client.retryBaseInterval * Math.pow(2, Math.min(10, client.retryAttempt)))
        onTriggered: {
            client.retryAttempt++
            if (client.jobId) client.checkStatus()
            else if (client.submissionPending) client.submitRequest()
        }
    }

    Timer {
        id: timeout
        interval: client.requestTimeout
        onTriggered: {
            client.cancelRequest()
            client.connectionFailed("Renderer request timed out. Retrying automatically.")
        }
    }

    function restoreCachedAssets() {
        if (!cachedDirectory || displayDirectory === cachedDirectory) return
        let saved = ({})
        try { saved = JSON.parse(cachedState || "{}") || ({}) } catch (failure) { /* legacy cache */ }
        if (saved.directory !== cachedDirectory) saved = ({})
        // The saved generation owns these settings, even if new settings are
        // being rendered. Restoring it never waits for an HTTP response.
        if (displayDirectory && !saved.generation) return
        displaySettings = Object.assign({}, saved.settings || requestSettings)
        displayRequestSettings = Object.assign({}, saved.request || saved.settings || requestSettings)
        displayPlaybackSettings = Object.assign({}, saved.playback || playbackSettings)
        displayManifest = saved.manifest || ({})
        displayImageCount = Number(saved.image_count) || (displaySettings.images || []).length
        displayFrameCount = Number(saved.frame_count) || Number(displaySettings.frames) || 16
        displayGeneration = saved.generation || cachedDirectory
        displayDirectory = cachedDirectory
        settingsPending = !displayMatchesRequest()
        if (!submissionPending && renderState !== "queued" && renderState !== "running") {
            cancelRequest()
            retry.stop()
            jobId = ""
            renderState = "ready"
            stage = "Ready"
            progress = 1
            resumeSavedStatus()
        }
    }

    function resumeSavedStatus() {
        // Observe saved generations without submitting a render or consulting
        // the current source revision. Imported/legacy paths need no service.
        const match = displayDirectory.match(/\/variants\/([a-f0-9]{24})\/[a-f0-9]{32}\/?$/)
        if (match && !jobId) jobId = match[1]
    }

    function displayMatchesRequest() {
        if (displayRequestSettings.size && JSON.stringify(displayRequestSettings.size)
                !== JSON.stringify([physicalWidth, physicalHeight])) return false
        if (displayRequestSettings.screen !== undefined && displayRequestSettings.screen !== screenName) return false
        return Object.keys(requestSettings).every(function(key) {
            let saved = displayRequestSettings[key]
            if (key === "perceptual_loss" && saved === undefined)
                saved = displaySettings.perceptual_loss !== undefined ? displaySettings.perceptual_loss
                    : displayRequestSettings.style === "headroom"
            return JSON.stringify(requestSettings[key]) === JSON.stringify(saved)
        })
    }

    function savePlaybackState() {
        // Saving updates the settings object's bindings. Wait until their
        // current evaluation has finished, and avoid saving identical state.
        if (renderState === "ready" && displayDirectory && publishedState !== cachedState)
            assetsReady()
    }

    function consume(job) {
        submissionPending = false
        stage = job.stage || ""
        error = job.error || ""
        progress = Math.max(0, Math.min(1, Number(job.progress) || 0))
        directory = job.directory || ""
        jobId = job.id || jobId
        renderState = job.state || "error"
        retryAttempt = 0
        retry.stop()
        if (renderState === "superseded") {
            jobId = ""
            renderState = displayDirectory ? "ready" : "idle"
            stage = "Render replaced. Select Render now to retry"
            return
        }
        if (renderState !== "ready" || !directory) return
        currentRequestForced = false
        const generation = String(job.generation || directory)
        if (displayDirectory === directory && displayGeneration === generation) {
            settingsPending = !displayMatchesRequest()
            const changed = !settingsPending
                && JSON.stringify(displayPlaybackSettings) !== JSON.stringify(playbackSettings)
            if (!settingsPending) displayPlaybackSettings = Object.assign({}, playbackSettings)
            if (changed) assetsReady()
            return
        }
        displaySettings = Object.assign({}, job.settings || pendingSettings || requestSettings)
        displayRequestSettings = Object.assign({}, Object.keys(pendingSettings).length
            ? pendingSettings : (job.settings || requestSettings))
        settingsPending = !displayMatchesRequest()
        if (!settingsPending || !displayDirectory)
            displayPlaybackSettings = Object.assign({}, playbackSettings)
        displayImageCount = Math.max(1, Number(job.image_count)
            || (displaySettings.images || []).length || 1)
        displayFrameCount = Math.max(2, Number(job.frame_count) || Number(displaySettings.frames) || 16)
        displayManifest = job.manifest || displayManifest
        displayGeneration = generation
        displayDirectory = directory
        assetsReady()
    }

    function cancelRequest() {
        requestSerial++
        const previous = activeRequest
        activeRequest = null
        requestInFlight = false
        timeout.stop()
        if (previous) previous.abort()
    }

    function connectionFailed(message) {
        renderState = "error"
        error = message
        retry.restart()
    }

    function sendRequest(method, path, payload, completion) {
        if (requestInFlight) return false
        const serial = ++requestSerial
        const xhr = new XMLHttpRequest()
        activeRequest = xhr
        requestInFlight = true
        xhr.open(method, serviceUrl + path, true)
        if (method === "POST") xhr.setRequestHeader("Content-Type", "application/json")
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || serial !== client.requestSerial) return
            client.timeoutFinished()
            if (xhr.status === 0 || xhr.status >= 500) {
                client.connectionFailed("Renderer service is unavailable. Retrying automatically.")
                return
            }
            try {
                const reply = JSON.parse(xhr.responseText)
                completion(xhr.status, reply)
            } catch (failure) {
                client.connectionFailed("Invalid response from renderer: " + String(failure))
            }
        }
        timeout.restart()
        xhr.send(payload === null ? null : JSON.stringify(payload))
        return true
    }

    function timeoutFinished() {
        timeout.stop()
        activeRequest = null
        requestInFlight = false
    }

    function start(force) {
        retry.stop()
        cancelRequest()
        jobId = ""
        submissionPending = false
        const request = JSON.parse(settingsKey)
        request.images = (request.images || []).map(path => String(path).trim()).filter(path => path.length)
        currentRequestForced = !!force
        if (!request.images.length) {
            renderState = "idle"
            stage = "Select an image"
            directory = ""
            displayDirectory = ""
            displayImageCount = 0
            displayManifest = ({})
            jobId = ""
            return
        }
        pendingSettings = Object.assign({}, request)
        submissionPending = true
        submitRequest()
    }

    function submitRequest() {
        if (!submissionPending) return
        const request = Object.assign({}, pendingSettings)
        request.force = currentRequestForced
        request.client_id = clientId
        renderState = "connecting"
        stage = "Connecting to renderer"
        error = ""
        sendRequest("POST", "/v1/render", request, function(status, reply) {
            if (status === 200) client.consume(reply)
            else { client.renderState = "error"; client.error = reply.error || "Render request failed" }
        })
    }

    function checkStatus() {
        if (!jobId || requestInFlight) return
        const expected = jobId
        sendRequest("GET", "/v1/status/" + encodeURIComponent(expected)
            + "?client_id=" + encodeURIComponent(clientId)
            + "&generation=" + encodeURIComponent(displayGeneration), null, function(status, reply) {
            if (expected !== client.jobId) return
            if (status === 200) client.consume(reply)
            else if (status === 404) {
                client.jobId = ""
                client.submissionPending = false
                retry.stop()
                client.renderState = client.displayDirectory ? "ready" : "idle"
                client.stage = "Saved render retained. Select Render now to update"
                client.error = ""
                if (client.autoRender) client.start(false)
            } else client.connectionFailed(reply.error || "Lost contact with renderer service.")
        })
    }
}
