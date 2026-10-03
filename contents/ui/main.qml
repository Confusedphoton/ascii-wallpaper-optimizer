/*
 * SPDX-FileCopyrightText: 2026 korvexx <korvexx.dev@proton.me>
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

import QtQuick
import "RenderSettings.js" as RenderSettings
import "Profiles.js" as Profiles
import QtQuick.Controls as Controls
import org.kde.plasma.plasmoid

WallpaperItem {
    id: root

    property string overrideProfile: "auto"
    property int timeRevision: 0
    readonly property var profiles: Profiles.read(wallpaper.configuration.Profiles, wallpaper.configuration)
    readonly property string profileChoices: JSON.stringify(profiles.map(profile => ({id: profile.id, name: profile.name})))
    readonly property string scheduledProfile: {
        timeRevision
        return Profiles.selected(profiles, wallpaper.configuration.ManualProfile,
            wallpaper.configuration.ScheduleEnabled, new Date())
    }
    readonly property string activeProfile: overrideProfile !== "auto" && profiles.some(profile => profile.id === overrideProfile)
        ? overrideProfile : scheduledProfile
    readonly property var activeSettings: (profiles.find(profile => profile.id === activeProfile) || profiles[0]).settings
    readonly property string mode: activeSettings.Mode
    readonly property var configImages: activeSettings.Images
    property var profilesRequest: null
    property string registeredChoices: ""
    onProfileChoicesChanged: Qt.callLater(publishProfiles)

    function publishProfiles() {
        if (profilesRequest || registeredChoices === profileChoices) return
        const choices = profileChoices
        const xhr = new XMLHttpRequest()
        profilesRequest = xhr
        xhr.open("POST", "http://127.0.0.1:48741/v1/profiles", true)
        xhr.setRequestHeader("Content-Type", "application/json")
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || root.profilesRequest !== xhr) return
            root.profilesRequest = null
            profileTimeout.stop()
            if (xhr.status === 200) root.registeredChoices = choices
        }
        profileTimeout.restart()
        xhr.send(choices)
    }
    property var overrideRequest: null

    RenderClient {
        id: renderer
        autoRender: true
        requestSettings: RenderSettings.request(
            root.mode === "Slideshow" ? Array.from(root.configImages) : Array.from(root.configImages).slice(0, 1), root.activeSettings)
        cachedDirectory: root.activeSettings.RenderDirectory
        cachedState: root.activeSettings.RenderedState
        playbackSettings: ({mode: root.mode, slideInterval: root.activeSettings.SlideInterval,
            transitionDuration: root.activeSettings.TransitionDuration})
        onAssetsReady: {
            const next = root.profiles.map(profile => profile.id === root.activeProfile
                ? Object.assign({}, profile, {settings: Object.assign({}, profile.settings,
                    {RenderDirectory: displayDirectory, RenderedState: publishedState})}) : profile)
            const serialized = JSON.stringify(next)
            if (wallpaper.configuration.Profiles !== serialized) {
                wallpaper.configuration.Profiles = serialized
                // Keep the original profiles readable by older plugin versions.
                if (root.activeProfile === "A" || root.activeProfile === "B") {
                    const suffix = root.activeProfile === "B" ? "B" : ""
                    wallpaper.configuration["RenderDirectory" + suffix] = displayDirectory
                    wallpaper.configuration["RenderedState" + suffix] = publishedState
                }
                wallpaper.configuration.writeConfig()
            }
        }
    }

    CacheClient {
        maximum: wallpaper.configuration.MaxCachedRenders || 20
        autoSync: true
    }

    PlaybackController {
        id: slideshow
        enabled: root.visible
        ready: shaderContainer.ready && shaderContainer.displayGeneration === renderer.displayGeneration
        loading: shaderContainer.loading
        slideshowEnabled: renderer.displayPlaybackSettings.mode === "Slideshow"
        manifest: renderer.displayManifest
        imageCount: renderer.displayImageCount
        frameCount: renderer.displayFrameCount
        generation: renderer.displayGeneration
        slideInterval: (renderer.displayPlaybackSettings.slideInterval || 10) * 1000
        transitionDuration: renderer.displayPlaybackSettings.transitionDuration || 1500
    }

    Timer {
        interval: 2000
        repeat: true
        running: true
        onTriggered: {
            root.timeRevision++
            root.fetchOverride()
        }
    }

    Timer {
        id: profileTimeout
        interval: 5000
        onTriggered: {
            const previous = root.profilesRequest
            root.profilesRequest = null
            if (previous) previous.abort()
        }
    }

    Timer {
        id: overrideTimeout
        interval: 5000
        onTriggered: {
            const previous = root.overrideRequest
            root.overrideRequest = null
            if (previous) previous.abort()
        }
    }

    function fetchOverride() {
        if (overrideRequest) return
        const xhr = new XMLHttpRequest()
        overrideRequest = xhr
        xhr.open("GET", "http://127.0.0.1:48741/v1/override", true)
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE || root.overrideRequest !== xhr) return
            root.overrideRequest = null
            overrideTimeout.stop()
            if (xhr.status === 200) {
                try {
                    const value = JSON.parse(xhr.responseText).profile
                    if (typeof value === "string") root.overrideProfile = value
                    root.publishProfiles()
                } catch (failure) { /* keep the last known override */ }
            }
        }
        overrideTimeout.restart()
        xhr.send()
    }

    AsciiShader {
        id: shaderContainer
        anchors.fill: parent
        
        renderDirectory: renderer.displayDirectory
        generation: renderer.displayGeneration
        manifest: renderer.displayManifest
        imageCount: renderer.displayImageCount
        currentIndex: slideshow.currentIndex
        nextIndex: slideshow.nextIndex
        transitionProgress: slideshow.transitionProgress
        transitionFrame: slideshow.transitionFrame
        transitionFrames: renderer.displayFrameCount
        loopFrame: slideshow.loopFrame
    }

    Rectangle {
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20
        width: Math.min(480, root.width - 40)
        height: statusColumn.implicitHeight + 24
        radius: 8
        color: "#b0000000"
        visible: !!shaderContainer.frameError || !shaderContainer.ready || renderer.renderState === "queued" || renderer.renderState === "running"
            || (renderer.renderState === "error" && !renderer.retrying)

        Column {
            id: statusColumn
            anchors.fill: parent
            anchors.margins: 12
            spacing: 6
            Controls.Label {
                width: parent.width
                visible: !!shaderContainer.frameError
                text: shaderContainer.frameError
                wrapMode: Text.WordWrap
                color: "white"
            }
            Controls.Label {
                width: parent.width
                wrapMode: Text.WordWrap
                color: "white"
                text: renderer.renderState === "error" ? renderer.error
                    : renderer.renderState === "idle" ? (root.configImages.length
                        ? "Preparing ASCII wallpaper" : "Select an image in wallpaper settings")
                    : renderer.stage || "Preparing ASCII wallpaper"
            }
            Controls.ProgressBar {
                width: parent.width
                visible: renderer.renderState === "running" || renderer.renderState === "queued" || renderer.renderState === "connecting"
                from: 0
                to: 1
                value: renderer.progress
                indeterminate: renderer.renderState === "connecting" || (renderer.renderState === "running" && renderer.progress === 0)
            }
        }
    }

    Component.onDestruction: {
        const previousProfiles = profilesRequest
        profilesRequest = null
        if (previousProfiles) previousProfiles.abort()
        const previous = overrideRequest
        overrideRequest = null
        if (previous) previous.abort()
    }

    Component.onCompleted: {
        root.loading = false;
        root.fetchOverride();
    }
}
