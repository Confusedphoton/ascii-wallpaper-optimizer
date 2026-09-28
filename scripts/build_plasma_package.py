"""Stage only the files needed by the KDE Plasma wallpaper package."""

from pathlib import Path
from shutil import copy2, copytree, rmtree


root = Path(__file__).resolve().parents[1]
package = root / "dist" / "plasma-wallpaper"
applet = root / "dist" / "plasma-profile-switch"
if package.is_symlink():
    raise RuntimeError(f"Refusing to replace a symlink: {package}")
if package.exists():
    rmtree(package)
package.mkdir(parents=True)
copy2(root / "metadata.json", package / "metadata.json")
copy2(root / "LICENSE.md", package / "LICENSE.md")
copytree(root / "contents", package / "contents")
print(package)
if applet.is_symlink():
    raise RuntimeError(f"Refusing to replace a symlink: {applet}")
if applet.exists():
    rmtree(applet)
applet.mkdir(parents=True)
copy2(root / "dock/metadata.json", applet / "metadata.json")
copy2(root / "LICENSE.md", applet / "LICENSE.md")
copytree(root / "dock/contents", applet / "contents")
print(applet)
