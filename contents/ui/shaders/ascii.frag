#version 440

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 qt_Matrix;
    float qt_Opacity;
    vec2 gridSize;
};

layout(binding = 1) uniform sampler2D glyphs;
layout(binding = 2) uniform sampler2D colors;
layout(binding = 3) uniform sampler2D atlas;
layout(binding = 4) uniform sampler2D metadata;
layout(binding = 5) uniform sampler2D backgrounds;

vec3 toLinear(vec3 value) {
    return mix(value / 12.92, pow((value + 0.055) / 1.055, vec3(2.4)),
               step(vec3(0.04045), value));
}

vec3 toSrgb(vec3 value) {
    return mix(value * 12.92, 1.055 * pow(value, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), value));
}

vec3 rgbToOklab(vec3 rgb) {
    vec3 lms = vec3(
        dot(rgb, vec3(0.4122214708, 0.5363325363, 0.0514459929)),
        dot(rgb, vec3(0.2119034982, 0.6806995451, 0.1073969566)),
        dot(rgb, vec3(0.0883024619, 0.2817188376, 0.6299787005)));
    lms = pow(max(lms, vec3(0.0)), vec3(1.0 / 3.0));
    return vec3(
        dot(lms, vec3(0.2104542553, 0.7936177850, -0.0040720468)),
        dot(lms, vec3(1.9779984951, -2.4285922050, 0.4505937099)),
        dot(lms, vec3(0.0259040371, 0.7827717662, -0.8086757660)));
}

vec3 oklabToRgb(vec3 lab) {
    vec3 lms = vec3(
        dot(lab, vec3(1.0, 0.3963377774, 0.2158037573)),
        dot(lab, vec3(1.0, -0.1055613458, -0.0638541728)),
        dot(lab, vec3(1.0, -0.0894841775, -1.2914855480)));
    lms = lms * lms * lms;
    return vec3(
        dot(lms, vec3(4.0767416621, -3.3077115913, 0.2309699292)),
        dot(lms, vec3(-1.2684380046, 2.6097574011, -0.3413193965)),
        dot(lms, vec3(-0.0041960863, -0.7034186147, 1.7076147010)));
}

vec3 gamutMap(vec3 lab) {
    lab.x = clamp(lab.x, 0.0, 1.0);
    vec3 rgb = oklabToRgb(lab);
    if (any(lessThan(rgb, vec3(-0.00001))) || any(greaterThan(rgb, vec3(1.00001)))) {
        float low = 0.0;
        float high = 1.0;
        for (int i = 0; i < 12; ++i) {
            float mid = 0.5 * (low + high);
            vec3 candidate = oklabToRgb(vec3(lab.x, lab.yz * mid));
            if (all(greaterThanEqual(candidate, vec3(-0.00001)))
                    && all(lessThanEqual(candidate, vec3(1.00001)))) {
                low = mid;
            } else {
                high = mid;
            }
        }
        rgb = oklabToRgb(vec3(lab.x, lab.yz * low));
    }
    return clamp(rgb, 0.0, 1.0);
}

vec3 displayTone(vec3 rgb, bool invertInk) {
    vec3 lab = rgbToOklab(rgb);
    float original = lab.x;
    float boosted = original + 2.0 * original * (1.0 - original) * (1.0 - original);
    lab.yz *= boosted / max(original, 0.000001);
    lab.x = boosted;
    vec3 displayed = gamutMap(lab);
    if (invertInk) {
        lab = rgbToOklab(displayed);
        lab.x = 1.0 - lab.x;
        displayed = gamutMap(lab);
    }
    return displayed;
}

void main() {
    vec3 widthData = texture(metadata, vec2(1.0 / 6.0, 0.5)).rgb;
    vec3 heightData = texture(metadata, vec2(3.0 / 6.0, 0.5)).rgb;
    vec3 countData = texture(metadata, vec2(5.0 / 6.0, 0.5)).rgb;
    vec2 outputSize = vec2(floor(widthData.r * 255.0 + 0.5) * 256.0 + floor(widthData.g * 255.0 + 0.5),
                           floor(heightData.r * 255.0 + 0.5) * 256.0 + floor(heightData.g * 255.0 + 0.5));
    vec2 cellSize = vec2(floor(widthData.b * 255.0 + 0.5), floor(heightData.b * 255.0 + 0.5));
    int atlasCount = int(floor(countData.r * 255.0 + 0.5)) + 1;
    bool invertInk = countData.g > 0.001;
    if (gridSize.x < 1.0 || gridSize.y < 1.0 || min(cellSize.x, cellSize.y) < 1.0
            || min(outputSize.x, outputSize.y) < 1.0 || atlasCount < 1) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0) * qt_Opacity;
        return;
    }
    vec2 position = qt_TexCoord0 * outputSize / cellSize;
    vec2 cell = min(floor(position), gridSize - 1.0);
    vec2 within = fract(position);
    vec2 mapUV = (cell + 0.5) / gridSize;
    int glyph = clamp(int(floor(texture(glyphs, mapUV).r * 255.0 + 0.5)), 0, atlasCount - 1);
    vec3 foreground = toLinear(texture(colors, mapUV).rgb);
    vec3 background = toLinear(texture(backgrounds, vec2(0.5)).rgb);
    vec2 atlasUV = vec2((float(glyph) + within.x) / float(atlasCount), within.y);
    float mask = texture(atlas, atlasUV).a;
    fragColor = vec4(toSrgb(displayTone(mix(background, foreground, mask), invertInk)), 1.0) * qt_Opacity;
}
