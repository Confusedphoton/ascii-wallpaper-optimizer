/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick

Item {
    id: player
    property bool playing: false
    property bool ready: false
    property bool loading: false
    property int frameCount: 120
    property int duration: 10000
    property string clipKey: ""
    property int frameIndex: 0
    property int ticks: 0
    property int reportedCycles: 0
    property real startedAt: 0
    property real pausedAt: 0
    property real pausedDuration: 0
    readonly property bool paused: loading || !ready
    readonly property bool timerRunning: frameTimer.running
    signal cycleCompleted(int cycles)

    onPlayingChanged: reset()
    onClipKeyChanged: reset()
    onFrameCountChanged: reset()
    onDurationChanged: reset()
    onPausedChanged: {
        if (!playing) return
        if (paused) {
            if (startedAt && !pausedAt) pausedAt = Date.now()
            frameTimer.stop()
        } else {
            if (pausedAt) pausedDuration += Date.now() - pausedAt
            pausedAt = 0
            Qt.callLater(schedule)
        }
    }

    function reset() {
        frameTimer.stop()
        frameIndex = 0
        ticks = 0
        reportedCycles = 0
        startedAt = 0
        pausedAt = 0
        pausedDuration = 0
        Qt.callLater(schedule)
    }

    function schedule() {
        if (!playing || paused || frameCount < 2) return
        const cycles = Math.floor(ticks / frameCount)
        if (frameIndex === 0 && cycles > reportedCycles) {
            reportedCycles = cycles
            cycleCompleted(cycles)
            if (!playing || paused) return
        }
        if (!startedAt) startedAt = Date.now()
        const deadline = (ticks + 1) * Math.max(1, duration) / frameCount
        frameTimer.interval = Math.max(1, Math.ceil(deadline - (Date.now() - startedAt - pausedDuration)))
        frameTimer.restart()
    }

    Timer {
        id: frameTimer
        onTriggered: {
            if (!player.playing || player.paused) return
            const deadline = player.startedAt + player.pausedDuration
                + (player.ticks + 1) * Math.max(1, player.duration) / player.frameCount
            // A blocked event loop shifts the clock too; present every frame
            // with its normal hold instead of bursting through late deadlines.
            const late = Date.now() - deadline
            if (late >= Math.max(1, player.duration) / player.frameCount)
                player.pausedDuration += late
            player.ticks++
            player.frameIndex = player.ticks % player.frameCount
            // Bindings update loading before the next deadline is scheduled.
            Qt.callLater(player.schedule)
        }
    }
}
