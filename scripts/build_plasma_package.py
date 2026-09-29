"""Stage Plasma packages and compile every shader from its current source."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns, rmtree, which


ROOT = Path(__file__).resolve().parents[1]


def find_qsb(explicit: str | None = None) -> str:
    if explicit or os.environ.get("QSB"):
        return explicit or os.environ["QSB"]
    executable = which("qsb") or which("qsb6")
    if executable:
        return executable
    qtpaths = which("qtpaths6")
    if qtpaths:
        result = subprocess.run([qtpaths, "--query", "QT_INSTALL_BINS"],
                                check=True, capture_output=True, text=True)
        candidate = Path(result.stdout.strip()) / "qsb"
        if candidate.is_file():
            return str(candidate)
    for directory in ("/usr/lib/qt6/bin", "/usr/lib64/qt6/bin", "/usr/lib/qt6/libexec"):
        candidate = Path(directory) / "qsb"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Qt 6 qsb is required to build the wallpaper. Install Qt Shader Tools or set QSB.")


def compile_shaders(contents: Path, qsb: str) -> None:
    for source in sorted((contents / "ui" / "shaders").iterdir()):
        if source.suffix not in (".vert", ".frag"):
            continue
        subprocess.run([qsb, "--glsl", "100 es,120,150", "--hlsl", "50", "--msl", "12",
                        "-o", str(source.with_suffix(source.suffix + ".qsb")), str(source)], check=True)


def build(root: Path = ROOT, destination: Path | None = None, qsb: str | None = None) -> tuple[Path, Path]:
    destination = destination or root / "dist"
    destination.mkdir(parents=True, exist_ok=True)
    compiler = find_qsb(qsb)
    # Compile before replacing an existing package, so a shader error leaves it usable.
    with tempfile.TemporaryDirectory(prefix="plasma-stage-", dir=destination) as temporary:
        stage = Path(temporary)
        wallpaper = stage / "plasma-wallpaper"
        wallpaper.mkdir()
        copy2(root / "metadata.json", wallpaper / "metadata.json")
        copy2(root / "LICENSE.md", wallpaper / "LICENSE.md")
        copytree(root / "contents", wallpaper / "contents", ignore=ignore_patterns("*.qsb"))
        compile_shaders(wallpaper / "contents", compiler)
        applet = stage / "plasma-profile-switch"
        applet.mkdir()
        copy2(root / "dock/metadata.json", applet / "metadata.json")
        copy2(root / "LICENSE.md", applet / "LICENSE.md")
        copytree(root / "dock/contents", applet / "contents")
        targets = []
        for package in (wallpaper, applet):
            target = destination / package.name
            if target.is_symlink():
                raise RuntimeError(f"Refusing to replace a symlink: {target}")
            if target.exists():
                rmtree(target)
            package.replace(target)
            targets.append(target)
        return tuple(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Package staging directory (default: dist)")
    parser.add_argument("--qsb", help="Qt 6 shader compiler executable")
    args = parser.parse_args()
    for package in build(destination=args.output, qsb=args.qsb):
        print(package)


if __name__ == "__main__":
    main()
