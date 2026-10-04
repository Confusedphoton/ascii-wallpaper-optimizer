/* SPDX-License-Identifier: AGPL-3.0-or-later */
import QtQuick

Item {
    id: controller
    property bool ready: false
    property bool loading: false
    property bool slideshowEnabled: false
    property int imageCount: 1
    property int frameCount: 16
    property int slideInterval: 10000
    property int transitionDuration: 1500
    property string generation: ""
    property var manifest: ({})
    readonly property int currentIndex: slideshow.currentIndex
    readonly property int nextIndex: slideshow.nextIndex
    readonly property int transitionFrame: slideshow.transitionFrame
    readonly property real transitionProgress: slideshow.transitionProgress
    readonly property var videoInfo: manifest.schema_version === 3 && manifest.video
        && manifest.video.path === "video" && manifest.video.frame_count >= 1
        && manifest.video.duration_ms > 0 ? manifest.video : null
    readonly property var loopInfo: videoInfo || (manifest.schema_version === 2
        ? (manifest.loops || ({}))[String(currentIndex).padStart(4, "0")] : null
    )
    readonly property bool hasLoop: !!videoInfo || (!!loopInfo && loopInfo.frame_count >= 4
        && loopInfo.duration_ms > 0 && loopInfo.path === "loops/" + String(currentIndex).padStart(4, "0"))
    readonly property int loopFrame: hasLoop && currentIndex === nextIndex ? loop.frameIndex : -1

    function reset() {
        slideshow.reset()
        loop.reset()
    }

    SlideshowPlayer {
        id: slideshow
        enabled: controller.enabled && controller.slideshowEnabled && !controller.videoInfo
        automaticAdvance: !controller.hasLoop
        ready: controller.ready
        loading: controller.loading
        imageCount: controller.imageCount
        frameCount: controller.frameCount
        generation: controller.generation
        slideInterval: controller.slideInterval
        transitionDuration: controller.transitionDuration
    }
    FlowLoopPlayer {
        id: loop
        playing: controller.enabled && controller.hasLoop && controller.currentIndex === controller.nextIndex
        ready: controller.ready
        loading: controller.loading
        frameCount: controller.loopInfo ? controller.loopInfo.frame_count : 120
        duration: controller.loopInfo ? controller.loopInfo.duration_ms : 10000
        frameDurations: controller.videoInfo ? controller.videoInfo.frame_durations_ms || [] : []
        clipKey: controller.generation + ":" + controller.currentIndex
        onCycleCompleted: cycles => {
            if (!controller.videoInfo && controller.slideshowEnabled && controller.imageCount > 1
                    && cycles * duration >= controller.slideInterval) slideshow.advance()
        }
    }
}
