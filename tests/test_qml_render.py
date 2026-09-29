"""Execute packaged shader binaries on an optional, headless Mesa EGL context.

This checks shader math and persisted PNG decoding, independently of Qt's RHI.
Qt scene graph lifecycle is covered in test_qml.py.
"""
from __future__ import annotations

import ctypes as C
from ctypes.util import find_library
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


@pytest.fixture(scope="module")
def packaged_glsl(tmp_path_factory):
    from scripts.build_plasma_package import build, find_qsb

    try:
        qsb = find_qsb()
    except RuntimeError as error:
        pytest.skip(str(error))
    directory = tmp_path_factory.mktemp("compiled-shader-parity")
    wallpaper, _ = build(destination=directory / "dist", qsb=qsb)
    sources = []
    for kind in ("vert", "frag"):
        source = directory / f"ascii.{kind}"
        binary = wallpaper / "contents/ui/shaders" / f"ascii.{kind}.qsb"
        subprocess.run([qsb, "-x", "glsl,100es", "-o", str(source), str(binary)],
                       check=True, capture_output=True, text=True)
        sources.append(str(source))
    return sources


@pytest.mark.parametrize("invert_ink", [False, True])
def test_packaged_shader_pixels_match_persisted_preview(tmp_path, packaged_glsl, invert_ink):
    from ascii_optimizer.assets import save_atlas, save_metadata
    from ascii_optimizer.cli import _save_grid
    from ascii_optimizer.optimizer import Grid
    from ascii_optimizer.tone import srgb_to_oklab

    # Exact atlas byte values avoid changing coverage during PNG serialization.
    mask = np.array([[0, 64, 128, 255], [255, 128, 64, 0],
                     [128, 255, 0, 64], [64, 0, 255, 128]], dtype=np.float32) / 255
    masks = np.stack((mask, mask[:, ::-1]))
    foreground = np.array([[[255, 0, 0], [0, 0, 255], [128, 128, 128], [0, 255, 0]],
                           [[255, 110, 0], [180, 20, 220], [255, 255, 255], [0, 0, 0]]],
                          dtype=np.float32) / 255
    grid = Grid(np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8),
                srgb_to_oklab(foreground), srgb_to_oklab(np.array([20, 30, 48], dtype=np.float32) / 255))
    persisted = _save_grid(grid, tmp_path, True, masks, (16, 8), invert_ink=invert_ink)
    save_atlas(masks, tmp_path / "atlas.png")
    save_metadata((16, 8), (4, 4), 2, tmp_path / "metadata.png", invert_ink=invert_ink)
    request = {"sources": packaged_glsl, "directory": str(tmp_path), "size": [16, 8],
               "cell": [4, 4], "grid": [4, 2], "count": 2, "invert": invert_ink,
               "background": persisted.background.tolist()}
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request))
    environment = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1", EGL_PLATFORM="surfaceless")
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    result = subprocess.run([sys.executable, str(Path(__file__)), str(request_file)],
                            env=environment, text=True, capture_output=True, timeout=20)
    if result.returncode == 77:
        pytest.skip(result.stdout.strip() or "Surfaceless EGL is unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    expected = np.asarray(Image.open(tmp_path / "preview.png"), dtype=np.int16)
    for name in ("uniforms", "legacy", "background-texture"):
        actual = np.fromfile(tmp_path / f"{name}.rgba", dtype=np.uint8).reshape(8, 16, 4)
        assert np.all(actual[..., 3] == 255)
        difference = np.abs(actual[..., :3].astype(np.int16) - expected)
        assert difference.max() <= 2, f"{name}: RGB byte error {difference.max()}"


def _render_surfaceless(request_file: Path) -> None:
    """Keep native graphics failure isolated from pytest in a subprocess."""
    request = json.loads(request_file.read_text())
    egl_name, gl_name = find_library("EGL"), find_library("GLESv2")
    if not egl_name or not gl_name:
        print("Mesa EGL and GLESv2 libraries are required for shader parity checks")
        raise SystemExit(77)
    egl, gl = C.CDLL(egl_name), C.CDLL(gl_name)
    I, U, F, P = C.c_int, C.c_uint, C.c_float, C.c_void_p

    def bind(library, name, result, *arguments):
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result
        return function

    platform = bind(egl, "eglGetPlatformDisplay", P, U, P, C.POINTER(I))
    initialize = bind(egl, "eglInitialize", U, P, C.POINTER(I), C.POINTER(I))
    choose = bind(egl, "eglChooseConfig", U, P, C.POINTER(I), C.POINTER(P), I, C.POINTER(I))
    bind_api = bind(egl, "eglBindAPI", U, U)
    create_surface = bind(egl, "eglCreatePbufferSurface", P, P, P, C.POINTER(I))
    create_context = bind(egl, "eglCreateContext", P, P, P, P, C.POINTER(I))
    make_current = bind(egl, "eglMakeCurrent", U, P, P, P, P)
    display = platform(0x31DD, None, None)  # EGL_PLATFORM_SURFACELESS_MESA
    major, minor = I(), I()
    config, count = P(), I()
    attributes = (I * 13)(0x3033, 1, 0x3040, 4, 0x3024, 8, 0x3023, 8, 0x3022, 8, 0x3021, 8, 0x3038)
    if (not initialize(display, C.byref(major), C.byref(minor))
            or not choose(display, attributes, C.byref(config), 1, C.byref(count)) or not count.value):
        print("Surfaceless Mesa EGL initialization is unavailable")
        raise SystemExit(77)
    width, height = request["size"]
    bind_api(0x30A0)  # EGL_OPENGL_ES_API
    surface = create_surface(display, config, (I * 5)(0x3057, width, 0x3056, height, 0x3038))
    context = create_context(display, config, None, (I * 3)(0x3098, 2, 0x3038))
    if not context or not surface or not make_current(display, surface, surface, context):
        print("Surfaceless OpenGL ES pbuffer context is unavailable")
        raise SystemExit(77)

    shader_create = bind(gl, "glCreateShader", U, U)
    shader_source = bind(gl, "glShaderSource", None, U, I, C.POINTER(C.c_char_p), C.POINTER(I))
    shader_compile = bind(gl, "glCompileShader", None, U)
    shader_status = bind(gl, "glGetShaderiv", None, U, U, C.POINTER(I))
    shader_log = bind(gl, "glGetShaderInfoLog", None, U, I, C.POINTER(I), C.c_char_p)
    program_create = bind(gl, "glCreateProgram", U)
    attach = bind(gl, "glAttachShader", None, U, U)
    link = bind(gl, "glLinkProgram", None, U)
    program_status = bind(gl, "glGetProgramiv", None, U, U, C.POINTER(I))
    program_log = bind(gl, "glGetProgramInfoLog", None, U, I, C.POINTER(I), C.c_char_p)
    use = bind(gl, "glUseProgram", None, U)
    uniform = bind(gl, "glGetUniformLocation", I, U, C.c_char_p)
    scalar = bind(gl, "glUniform1f", None, I, F)
    integer = bind(gl, "glUniform1i", None, I, I)
    vec2 = bind(gl, "glUniform2f", None, I, F, F)
    vec3 = bind(gl, "glUniform3f", None, I, F, F, F)
    matrix = bind(gl, "glUniformMatrix4fv", None, I, I, U, C.POINTER(F))
    attribute = bind(gl, "glGetAttribLocation", I, U, C.c_char_p)
    enable_attribute = bind(gl, "glEnableVertexAttribArray", None, U)
    attribute_pointer = bind(gl, "glVertexAttribPointer", None, U, I, U, U, I, P)
    generate_buffers = bind(gl, "glGenBuffers", None, I, C.POINTER(U))
    buffer_bind = bind(gl, "glBindBuffer", None, U, U)
    buffer_data = bind(gl, "glBufferData", None, U, C.c_ssize_t, P, U)
    generate_textures = bind(gl, "glGenTextures", None, I, C.POINTER(U))
    active_texture = bind(gl, "glActiveTexture", None, U)
    texture_bind = bind(gl, "glBindTexture", None, U, U)
    texture_parameter = bind(gl, "glTexParameteri", None, U, U, I)
    texture_data = bind(gl, "glTexImage2D", None, U, I, I, I, I, I, U, U, P)
    viewport = bind(gl, "glViewport", None, I, I, I, I)
    draw = bind(gl, "glDrawArrays", None, U, I, I)
    read = bind(gl, "glReadPixels", None, I, I, I, I, U, U, P)
    error = bind(gl, "glGetError", U)
    program = program_create()
    prefixes = []
    for source_path, kind in zip(request["sources"], (0x8B31, 0x8B30)):
        source = Path(source_path).read_text()
        prefixes.extend(re.findall(r"uniform buf (\w+);", source))
        shader = shader_create(kind)
        pointer = C.c_char_p(source.encode())
        shader_source(shader, 1, C.byref(pointer), None)
        shader_compile(shader)
        status = I()
        shader_status(shader, 0x8B81, C.byref(status))
        if not status.value:
            log = C.create_string_buffer(8192)
            shader_log(shader, len(log), None, log)
            raise AssertionError(log.value.decode())
        attach(program, shader)
    link(program)
    status = I()
    program_status(program, 0x8B82, C.byref(status))
    if not status.value:
        log = C.create_string_buffer(8192)
        program_log(program, len(log), None, log)
        raise AssertionError(log.value.decode())
    use(program)

    def set_uniform(name, function, *values):
        for prefix in set(prefixes):
            location = uniform(program, f"{prefix}.{name}".encode())
            if location >= 0:
                function(location, *values)

    identity = (F * 16)(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    set_uniform("qt_Matrix", matrix, 1, 0, identity)
    set_uniform("qt_Opacity", scalar, 1)
    for name, values in (("gridSize", request["grid"]), ("outputSize", request["size"]),
                         ("cellSize", request["cell"])):
        set_uniform(name, vec2, *values)
    set_uniform("atlasCount", scalar, request["count"])
    set_uniform("invertInk", scalar, int(request["invert"]))
    set_uniform("backgroundLab", vec3, *request["background"])
    vertices = (F * 16)(-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1)
    buffer = U()
    generate_buffers(1, C.byref(buffer))
    buffer_bind(0x8892, buffer)
    buffer_data(0x8892, C.sizeof(vertices), vertices, 0x88E4)
    for name, offset in (("qt_Vertex", 0), ("qt_MultiTexCoord0", 8)):
        location = attribute(program, name.encode())
        enable_attribute(location)
        attribute_pointer(location, 2, 0x1406, 0, 16, P(offset))
    directory = Path(request["directory"])
    textures = (U * 5)()
    generate_textures(5, textures)
    for unit, name in enumerate(("glyphs", "colors", "backgrounds", "atlas", "metadata")):
        filename = {"glyphs": "glyph", "colors": "color", "backgrounds": "background"}.get(name, name)
        image = Image.open(directory / f"{filename}.png").convert("RGBA")
        pixels = C.create_string_buffer(image.tobytes())
        active_texture(0x84C0 + unit)
        texture_bind(0x0DE1, textures[unit])
        for parameter, value in ((0x2800, 0x2600), (0x2801, 0x2600), (0x2802, 0x812F), (0x2803, 0x812F)):
            texture_parameter(0x0DE1, parameter, value)
        texture_data(0x0DE1, 0, 0x1908, image.width, image.height, 0, 0x1908, 0x1401, pixels)
        integer(uniform(program, name.encode()), unit)
    viewport(0, 0, width, height)
    for name, metadata, background in (("uniforms", 1, 1), ("legacy", 0, 0), ("background-texture", 1, 0)):
        set_uniform("metadataAvailable", scalar, metadata)
        set_uniform("backgroundAvailable", scalar, background)
        draw(0x0005, 0, 4)
        pixels = (C.c_ubyte * (width * height * 4))()
        read(0, 0, width, height, 0x1908, 0x1401, pixels)
        assert error() == 0, "OpenGL error while rendering packaged shaders"
        (directory / f"{name}.rgba").write_bytes(bytes(pixels))
    make_current(display, None, None, None)
    bind(egl, "eglDestroyContext", U, P, P)(display, context)
    bind(egl, "eglDestroySurface", U, P, P)(display, surface)
    bind(egl, "eglTerminate", U, P)(display)


if __name__ == "__main__":
    _render_surfaceless(Path(sys.argv[1]))
