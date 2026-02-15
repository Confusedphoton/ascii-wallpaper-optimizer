#version 440

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 qt_Matrix;
    float qt_Opacity;
    vec2 charSize;
    vec2 viewSize;
    int atlasCount;
    int isGrayscaled;
    float invAtlasCount;
};

layout(binding = 1) uniform sampler2D src;
layout(binding = 2) uniform sampler2D atlas;

void main() {
    vec2 grid = viewSize / charSize;
    vec2 cellUV = (floor(qt_TexCoord0 * grid) + 0.5) / grid;
    vec2 localUV = fract(qt_TexCoord0 * grid);

    vec4 color = texture(src, cellUV);

    float brightness = clamp(dot(color.rgb, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
    vec3 finalRGB = mix(color.rgb, vec3(brightness), float(isGrayscaled));

    float charIdx = floor(brightness * float(atlasCount - 1));
    float centeredX = 0.25 + (localUV.x * 0.5);

    float mask = texture(atlas, vec2((charIdx + centeredX) * invAtlasCount, localUV.y)).a;

    fragColor = vec4(finalRGB * mask, color.a * qt_Opacity);
}
