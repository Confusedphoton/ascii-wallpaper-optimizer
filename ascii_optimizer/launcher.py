"""Launch numerical tools without exposing unrelated Linux NPU devices.

ROCm 7.2 discovers /dev/accel even for GPU-only workloads. An unhealthy
amdxdna driver can block that discovery in an uninterruptible kernel wait.
Hide NPU devices in this process's mount namespace; keep /dev/dri and /dev/kfd
available so GPU solving still works. The service itself needs no namespace.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

ACCEL_PATH = Path("/dev/accel")


def numerical_command(arguments: list[str]) -> list[str]:
    command = [sys.executable, *arguments]
    if sys.platform.startswith("linux") and any(ACCEL_PATH.glob("accel*")):
        wrapper = shutil.which("bwrap")
        if wrapper is None:
            raise RuntimeError("Install bubblewrap to isolate NPU devices before starting the optimizer.")
        # A plain bind of /dev makes character devices unusable in bubblewrap.
        command = [wrapper, "--die-with-parent", "--bind", "/", "/",
                   "--dev-bind", "/dev", "/dev", "--tmpfs", "/dev/accel", *command]
    return command


def _exec(arguments: list[str]) -> None:
    try:
        command = numerical_command(arguments)
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
    os.execvp(command[0], command)


def render_main() -> None:
    _exec(["-m", "ascii_optimizer.cli", *sys.argv[1:]])


def main() -> None:
    _exec(sys.argv[1:])


if __name__ == "__main__":
    main()
