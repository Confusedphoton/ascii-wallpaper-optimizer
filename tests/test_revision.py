"""Cache identities follow renderer contents rather than just release numbers."""
from pathlib import Path

from ascii_optimizer import revision, service
from ascii_optimizer.cli import make_parser, build
from test_service import publish
from PIL import Image


def test_renderer_revision_uses_contents_and_is_portable(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for directory in (left, right):
        (directory / "tone.py").write_text("color transform v1")
        (directory / "revision.py").write_text("revision function")
    monkeypatch.setattr(revision, "RENDER_MODULES", ("tone.py", "revision.py"))
    monkeypatch.setattr(revision, "__file__", str(left / "revision.py"))
    first = revision.render_revision()
    monkeypatch.setattr(revision, "__file__", str(right / "revision.py"))
    assert revision.render_revision() == first
    (right / "tone.py").write_text("color transform v2")
    assert revision.render_revision() != first


def test_changed_renderer_rebuilds_cached_service_variant(tmp_path, monkeypatch):
    image, font = tmp_path / "source.png", tmp_path / "font.ttf"
    Image.new("RGB", (8, 8)).save(image)
    font.write_bytes(b"font")
    monkeypatch.setattr(service, "_screen_size", lambda name, size: tuple(size))
    monkeypatch.setattr(service, "render_revision", lambda: "old-code")
    settings = {"images": [str(image)], "font": str(font), "size": [8, 8]}
    queue = service.RenderQueue(tmp_path / "cache", start_worker=False)
    first = queue.submit(settings)
    publish(queue, first)
    assert queue.submit(settings)["state"] == "ready"
    monkeypatch.setattr(service, "render_revision", lambda: "fixed-code")
    updated = queue.submit(settings)
    assert updated["state"] == "queued"
    assert updated["id"] != first["id"]
    assert updated["directory"] != first["directory"]


def test_changed_renderer_rebuilds_still_cache(tmp_path, monkeypatch, capsys):
    import ascii_optimizer.cli as cli

    image = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (170, 20, 80)).save(image)
    args = make_parser().parse_args([str(image), "--output", str(tmp_path / "output"),
        "--still-cache", str(tmp_path / "stills"), "--size", "8x8", "--cell", "4x4",
        "--chars", " .#", "--steps", "1", "--device", "cpu",
        "--gradient-loss-strength", "0", "--glyph-luminance-bonus", "0"])
    monkeypatch.setattr(cli, "render_revision", lambda: "old-code")
    build(args)
    capsys.readouterr()
    args.output = tmp_path / "second-output"
    build(args)
    assert "Reusing cached still" in capsys.readouterr().out
    monkeypatch.setattr(cli, "render_revision", lambda: "fixed-code")
    build(args)
    output = capsys.readouterr().out
    assert "Optimizing still" in output
    assert "Reusing" not in output
    assert len(list((tmp_path / "stills").iterdir())) == 2
