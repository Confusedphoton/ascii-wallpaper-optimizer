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

vec3 sampleBackground(vec2 position) {
    vec2 source = position - 0.5;
    ivec2 low = ivec2(floor(source));
    ivec2 high = low + ivec2(1);
    ivec2 limit = ivec2(gridSize) - ivec2(1);
    low = clamp(low, ivec2(0), limit);
    high = clamp(high, ivec2(0), limit);
    vec2 blend = fract(source);
    vec3 top = mix(toLinear(texture(backgrounds, (vec2(low.x, low.y) + 0.5) / gridSize).rgb),
                   toLinear(texture(backgrounds, (vec2(high.x, low.y) + 0.5) / gridSize).rgb), blend.x);
    vec3 bottom = mix(toLinear(texture(backgrounds, (vec2(low.x, high.y) + 0.5) / gridSize).rgb),
                      toLinear(texture(backgrounds, (vec2(high.x, high.y) + 0.5) / gridSize).rgb), blend.x);
    return mix(top, bottom, blend.y);
}

void main() {
    vec3 widthData = texture(metadata, vec2(1.0 / 6.0, 0.5)).rgb;
    vec3 heightData = texture(metadata, vec2(3.0 / 6.0, 0.5)).rgb;
    vec3 countData = texture(metadata, vec2(5.0 / 6.0, 0.5)).rgb;
    vec2 outputSize = vec2(floor(widthData.r * 255.0 + 0.5) * 256.0 + floor(widthData.g * 255.0 + 0.5),
                           floor(heightData.r * 255.0 + 0.5) * 256.0 + floor(heightData.g * 255.0 + 0.5));
    vec2 cellSize = vec2(floor(widthData.b * 255.0 + 0.5), floor(heightData.b * 255.0 + 0.5));
    int atlasCount = int(floor(countData.r * 255.0 + 0.5)) + 1;
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
    vec3 background = sampleBackground(position);
    vec2 atlasUV = vec2((float(glyph) + within.x) / float(atlasCount), within.y);
    float mask = texture(atlas, atlasUV).a;
    fragColor = vec4(toSrgb(mix(background, foreground, mask)), 1.0) * qt_Opacity;
}
