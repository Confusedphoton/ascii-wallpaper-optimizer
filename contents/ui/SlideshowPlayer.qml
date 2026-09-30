/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick

Item {
    id: player
    property bool ready: false
    property bool loading: false
    property int imageCount: 0
    property int slideInterval: 10000
    property int transitionDuration: 1500
    property int frameCount: 16
    property string generation: ""
    property int currentIndex: 0
    property int nextIndex: 0
    property int transitionFrame: 0
    readonly property real transitionProgress: transitionFrame / (playbackFrameCount - 1)
    readonly property bool timerRunning: slideTimer.running
    readonly property bool animating: transitionActive
    readonly property bool paused: transitionActive && (loading || !ready)
    property bool resetting: false
    property bool transitionActive: false
    property int playbackFrameCount: 16
    property int playbackDuration: 1500
    property real startedAt: 0
    property real pausedAt: 0
    property real pausedDuration: 0

    onGenerationChanged: reset()
    onEnabledChanged: if (!enabled) reset()
    onFrameCountChanged: if (animating) reset()
    onImageCountChanged: {
        if (currentIndex >= imageCount || nextIndex >= imageCount) reset()
    }
    onPausedChanged: {
        if (resetting || !animating) return
        if (paused) {
            pausedAt = Date.now()
            frameTimer.stop()
        } else {
            pausedDuration += Date.now() - pausedAt
            pausedAt = 0
            scheduleFrame()
        }
    }

    function reset() {
        resetting = true
        frameTimer.stop()
        transitionActive = false
        currentIndex = 0
        nextIndex = 0
        transitionFrame = 0
        pausedAt = 0
        pausedDuration = 0
        resetting = false
    }

    function advance() {
        if (!enabled || !ready || loading || imageCount < 2 || currentIndex !== nextIndex) return
        playbackFrameCount = Math.max(2, frameCount)
        playbackDuration = Math.max(1, transitionDuration)
        transitionFrame = 0
        startedAt = Date.now()
        pausedDuration = 0
        pausedAt = 0
        nextIndex = (currentIndex + 1) % imageCount
        transitionActive = true
        if (paused) pausedAt = Date.now()
        scheduleFrame()
    }

    function scheduleFrame() {
        if (!animating || paused || resetting) return
        if (transitionFrame === playbackFrameCount - 1) {
            transitionActive = false
            currentIndex = nextIndex
            transitionFrame = 0
            return
        }
        // Absolute deadlines avoid accumulating timer rounding over long holds.
        // Loading pauses the clock, and even late deadlines present each frame.
        const deadline = (transitionFrame + 1) * playbackDuration / (playbackFrameCount - 1)
        const elapsed = Date.now() - startedAt - pausedDuration
        frameTimer.interval = Math.max(1, Math.ceil(deadline - elapsed))
        frameTimer.restart()
    }

    Timer {
        id: slideTimer
        interval: Math.max(1, player.slideInterval)
        repeat: true
        running: player.enabled && player.ready && !player.loading && !player.resetting
            && player.imageCount > 1 && player.currentIndex === player.nextIndex
        onTriggered: player.advance()
    }

    Timer {
        id: frameTimer
        onTriggered: {
            if (!player.animating || player.paused) return
            player.transitionFrame++
            player.scheduleFrame()
        }
    }
}
