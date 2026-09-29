/* SPDX-License-Identifier: AGPL-3.0-or-later */

import QtQuick

Item {
    id: player
    property bool ready: false
    property bool loading: false
    property int imageCount: 0
    property int slideInterval: 10000
    property int transitionDuration: 1500
    property string generation: ""
    property int currentIndex: 0
    property int nextIndex: 0
    property real transitionProgress: 0
    readonly property bool timerRunning: slideTimer.running
    readonly property bool animating: transition.running
    readonly property bool paused: transition.paused
    property bool resetting: false

    onGenerationChanged: reset()
    onEnabledChanged: if (!enabled) reset()
    onImageCountChanged: {
        if (currentIndex >= imageCount || nextIndex >= imageCount) reset()
    }

    function reset() {
        resetting = true
        transition.stop()
        currentIndex = 0
        nextIndex = 0
        transitionProgress = 0
        resetting = false
    }

    function advance() {
        if (!enabled || !ready || loading || imageCount < 2 || currentIndex !== nextIndex) return
        nextIndex = (currentIndex + 1) % imageCount
        transition.start()
    }

    Timer {
        id: slideTimer
        interval: Math.max(1, player.slideInterval)
        repeat: true
        running: player.enabled && player.ready && !player.loading && !player.resetting
            && player.imageCount > 1 && player.currentIndex === player.nextIndex
        onTriggered: player.advance()
    }

    NumberAnimation {
        id: transition
        target: player
        property: "transitionProgress"
        from: 0
        to: 1
        duration: Math.max(1, player.transitionDuration)
        easing.type: Easing.Linear
        paused: player.loading && running
        onFinished: {
            player.currentIndex = player.nextIndex
            player.transitionProgress = 0
        }
    }
}
