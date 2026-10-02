/* SPDX-License-Identifier: AGPL-3.0-or-later */
.pragma library

const standardChars = " .´:rszunvcxLOGDQ0KNMW&#\\/=^|[]{}"
const structureChars = " .´:-_/\\=^|+rszunvcxLOGDQ0KNMW&#[]{}"
const defaults = {
    Mode: "Image", Images: [], FontFamily: "Monospace", AsciiChars: Array.from(standardChars),
    Detail: 1, Grayscale: false, SlideInterval: 10, TransitionFrames: 16,
    TransitionDuration: 1500, TransitionPace: "ease-in-out", TransitionField: "blue-noise",
    TransitionSpread: 0.65, Dither: 0.005, GradientLossStrength: 1.5,
    GlyphLuminanceBonus: 0.12, GlyphEntropyStrength: 0.01, StructureDither: 0.06,
    InvertInk: false, BackgroundMode: 0, BackgroundColor: "#202028",
    RenderStyle: "headroom", RenderDirectory: "", RenderedState: ""
}

function normalizedSettings(settings) {
    const result = Object.assign({}, defaults, settings)
    if (result.RenderStyle === "headroom" && result.PerceptualLoss === false) result.RenderStyle = "classic"
    delete result.PerceptualLoss
    result.Images = Array.from(result.Images || [])
    let chars = Array.from(result.AsciiChars || defaults.AsciiChars)
    if (chars[0] === "\\s") chars[0] = " "
    const previousDefaults = [
        " .´:rszunvcxLOGDQ0KNMW&@#\\/=^|[]{}", " .´:rszunvcxLOGDQ0KNMW&@\\/=^|[]{}",
        " .´:rszunvcxLOGDQ0KNMW&@\\=^|[]{}", " .´:rszunvcxLOGDQ0KNMW&@",
        " .´:rszunvcxLOGDQ0KNMW&@\\=^", " .´:-_/\\=^|+rszunvcxLOGDQ0KNMW&@[]{}",
        " .´:-_/|+rszunvcxLOGDQ0KNMW&@[]{}"
    ]
    const current = chars.join("")
    if (previousDefaults.includes(current)
            || (result.RenderStyle === "structure" && current === standardChars)
            || (result.RenderStyle !== "structure" && current === structureChars))
        chars = Array.from(result.RenderStyle === "structure" ? structureChars : standardChars)
    result.AsciiChars = chars
    return result
}

function decode(serialized) {
    try {
        const profiles = JSON.parse(serialized || "[]")
        const ids = new Set()
        if (!Array.isArray(profiles)) return []
        return profiles.filter(profile => {
            if (!profile || !/^[A-Za-z0-9_.-]{1,64}$/.test(profile.id)
                    || profile.id === "auto" || ids.has(profile.id)) return false
            ids.add(profile.id)
            return true
        }).map(profile => ({id: profile.id, name: String(profile.name || profile.id),
            startTime: minute(profile.startTime) >= 0 ? profile.startTime : "00:00",
            settings: normalizedSettings(profile.settings)}))
    } catch (failure) { return [] }
}

function fromLegacy(config) {
    return ["A", "B"].map(id => {
        const settings = {}
        for (const name of Object.keys(defaults)) {
            const suffix = id === "B" && (config.ProfileSettingsInitialized
                || name === "Mode" || name === "Images") ? "B" : ""
            if (config[name + suffix] !== undefined) settings[name] = config[name + suffix]
        }
        const objectiveSuffix = id === "B" && config.ProfileSettingsInitialized ? "B" : ""
        settings.PerceptualLoss = config["PerceptualLoss" + objectiveSuffix]
        if (id === "B" && !config.ProfileSettingsInitialized) {
            settings.RenderDirectory = ""
            settings.RenderedState = ""
        }
        if (id === "A" && config.Image && !Array.from(settings.Images || []).includes(config.Image))
            settings.Images = Array.from(settings.Images || []).concat([config.Image])
        return {id: id, name: "Profile " + id,
            startTime: config["Profile" + id + "Start"] || (id === "A" ? "07:00" : "19:00"),
            settings: normalizedSettings(settings)}
    })
}

function read(serialized, config) {
    const profiles = decode(serialized)
    return profiles.length ? profiles : fromLegacy(config)
}

function minute(value) {
    if (!/^([01][0-9]|2[0-3]):[0-5][0-9]$/.test(String(value))) return -1
    const parts = value.split(":").map(Number)
    return parts[0] * 60 + parts[1]
}

function selected(profiles, manual, scheduled, now) {
    if (!profiles.length) return "A"
    if (!scheduled) return profiles.some(profile => profile.id === manual) ? manual : profiles[0].id
    const current = now.getHours() * 60 + now.getMinutes()
    const ordered = profiles.filter(profile => minute(profile.startTime) >= 0)
        .slice().sort((a, b) => minute(a.startTime) - minute(b.startTime))
    if (!ordered.length) return profiles[0].id
    let active = ordered[ordered.length - 1]
    for (const profile of ordered) {
        if (minute(profile.startTime) > current) break
        active = profile
    }
    return active.id
}

function availableStart(profiles) {
    const used = new Set(profiles.map(profile => minute(profile.startTime)))
    for (let value = 0; value < 1440; value += 60) {
        if (!used.has(value)) return String(Math.floor(value / 60)).padStart(2, "0") + ":00"
    }
    for (let value = 0; value < 1440; value++) {
        if (!used.has(value)) return String(Math.floor(value / 60)).padStart(2, "0") + ":"
            + String(value % 60).padStart(2, "0")
    }
    return "00:00"
}
