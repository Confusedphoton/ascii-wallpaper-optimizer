/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick

Item {
    id: player
    property bool playing: false
    property bool ready: false
    property bool loading: false
    property bool followTimeline: false
    property int frameCount: 120
    property real duration: 10000
    property var frameDurations: []
    readonly property var frameOffsets: {
        if (!frameDurations || frameDurations.length !== frameCount) return []
        const offsets = [0]
        for (const hold of frameDurations) {
            if (!Number.isFinite(Number(hold)) || Number(hold) <= 0) return []
            offsets.push(offsets[offsets.length - 1] + Number(hold))
        }
        return offsets
    }
    property string clipKey: ""
    property int frameIndex: 0
    property int ticks: 0
    property int reportedCycles: 0
    property real startedAt: 0
    property real pausedAt: 0
    property real pausedDuration: 0
    readonly property bool paused: followTimeline ? !startedAt && !ready : loading || !ready
    readonly property bool timerRunning: frameTimer.running
    signal cycleCompleted(int cycles)

    onPlayingChanged: reset()
    onFollowTimelineChanged: reset()
    onClipKeyChanged: reset()
    onFrameCountChanged: reset()
    onDurationChanged: reset()
    onFrameOffsetsChanged: reset()
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
        if (followTimeline) {
            scheduleVideo()
            return
        }
        const cycles = Math.floor(ticks / frameCount)
        if (frameIndex === 0 && cycles > reportedCycles) {
            reportedCycles = cycles
            cycleCompleted(cycles)
            if (!playing || paused) return
        }
        if (!startedAt) startedAt = Date.now()
        const deadline = nextDeadline()
        frameTimer.interval = Math.max(1, Math.ceil(deadline - (Date.now() - startedAt - pausedDuration)))
        frameTimer.restart()
    }

    function scheduleVideo() {
        if (!startedAt) startedAt = Date.now()
        const elapsed = Math.max(0, Date.now() - startedAt)
        const cycleDuration = frameOffsets.length ? frameOffsets[frameCount] : duration
        const phase = elapsed % cycleDuration
        let deadline
        if (frameOffsets.length) {
            let low = 0
            let high = frameCount
            while (low + 1 < high) {
                const middle = Math.floor((low + high) / 2)
                if (frameOffsets[middle] <= phase) low = middle
                else high = middle
            }
            frameIndex = low
            deadline = frameOffsets[low + 1]
        } else {
            const hold = cycleDuration / frameCount
            frameIndex = Math.min(frameCount - 1, Math.floor(phase / hold))
            deadline = (frameIndex + 1) * hold
        }
        frameTimer.interval = Math.max(1, Math.min(2147483647, Math.ceil(deadline - phase)))
        frameTimer.restart()
    }

    function nextDeadline() {
        return frameOffsets.length ? Math.floor(ticks / frameCount) * frameOffsets[frameCount]
            + frameOffsets[frameIndex + 1] : (ticks + 1) * Math.max(1, duration) / frameCount
    }

    function currentHold() {
        return frameOffsets.length ? frameOffsets[frameIndex + 1] - frameOffsets[frameIndex]
            : Math.max(1, duration) / frameCount
    }

    Timer {
        id: frameTimer
        onTriggered: {
            if (!player.playing || player.paused) return
            if (player.followTimeline) {
                player.scheduleVideo()
                return
            }
            const deadline = player.startedAt + player.pausedDuration + player.nextDeadline()
            // A blocked event loop shifts the clock too; present every frame
            // with its normal hold instead of bursting through late deadlines.
            const late = Date.now() - deadline
            if (late >= player.currentHold())
                player.pausedDuration += late
            player.ticks++
            player.frameIndex = player.ticks % player.frameCount
            // Bindings update loading before the next deadline is scheduled.
            Qt.callLater(player.schedule)
        }
    }
}
