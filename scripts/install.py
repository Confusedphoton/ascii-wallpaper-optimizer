"""Install the Plasma package and its local asynchronous render service."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = json.loads((ROOT / "metadata.json").read_text())["KPlugin"]["Id"]
PACKAGE = ROOT / "dist" / "plasma-wallpaper"
APPLET = ROOT / "dist" / "plasma-profile-switch"
UNIT = Path.home() / ".config/systemd/user/ascii-wallpaper-renderer.service"


def run(*command: str) -> None:
    print("+", *command, flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        python = Path(sys.executable)
    if not (ROOT / "ascii_optimizer/service.py").is_file():
        raise SystemExit("Renderer source is missing.")
    run(str(python), "-c", "import numpy, PIL, torch")
    run(str(python), str(ROOT / "scripts/build_plasma_package.py"))
    installed = Path.home() / ".local/share/plasma/wallpapers" / PACKAGE_ID
    action = "--upgrade" if installed.exists() else "--install"
    run("kpackagetool6", "--type", "Plasma/Wallpaper", action, str(PACKAGE))
    applet_installed = Path.home() / ".local/share/plasma/plasmoids/org.asciiwallpaper.profile-switch"
    applet_action = "--upgrade" if applet_installed.exists() else "--install"
    run("kpackagetool6", "--type", "Plasma/Applet", applet_action, str(APPLET))

    UNIT.parent.mkdir(parents=True, exist_ok=True)
    UNIT.write_text("\n".join([
        "[Unit]", "Description=ASCII Wallpaper background renderer", "After=graphical-session.target", "",
        "[Service]", "Type=simple", f"WorkingDirectory={ROOT}",
        f"ExecStart={json.dumps(str(python))} -m ascii_optimizer.service", "Restart=on-failure", "",
        "[Install]", "WantedBy=default.target", "",
    ]))
    run("systemctl", "--user", "daemon-reload")
    run("systemctl", "--user", "enable", "--now", "ascii-wallpaper-renderer.service")
    run("systemctl", "--user", "restart", "ascii-wallpaper-renderer.service")
    run("kbuildsycoca6", "--noincremental")
    # Plasma keeps the previous QML and compiled shader in a running wallpaper
    # instance even after kpackagetool upgrades the files on disk.
    if action == "--upgrade":
        run("systemctl", "--user", "try-restart", "plasma-plasmashell.service")
    print("Installed. Reopen wallpaper settings, then add ASCII Wallpaper Profile to your panel or dock.")


if __name__ == "__main__":
    main()
