#!/usr/bin/env python3
"""Install the project with a PyTorch build selected for the available GPU."""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def detect_gpu_vendors() -> set[str]:
    """Read Linux PCI display device IDs without requiring lspci."""
    if not sys.platform.startswith("linux"):
        return set()

    vendors: set[str] = set()
    for device in glob.glob("/sys/bus/pci/devices/*"):
        try:
            vendor = (Path(device) / "vendor").read_text().strip().lower()
            device_class = int((Path(device) / "class").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        if device_class >> 16 != 0x03:
            continue
        if vendor == "0x10de":
            vendors.add("NVIDIA")
        elif vendor == "0x1002":
            vendors.add("AMD")
    return vendors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda", "rocm"),
        default="auto",
        help="override GPU detection (ROCm is supported on Linux)",
    )
    args = parser.parse_args()

    if args.backend == "auto":
        vendors = detect_gpu_vendors()
        if "NVIDIA" in vendors:
            backend = "auto"
            print(
                f"Detected {', '.join(sorted(vendors))} GPU(s); asking uv to choose a compatible PyTorch build.",
                flush=True,
            )
        elif "AMD" in vendors:
            # Driver/library discovery can miss AMD GPUs even when /dev/kfd
            # works. PCI detection must not silently install CPU-only torch.
            backend = "rocm7.2"
            print("Detected AMD GPU; selecting ROCm 7.2 PyTorch.", flush=True)
        else:
            backend = "cpu"
            print("No AMD or NVIDIA GPU detected; selecting CPU PyTorch.", flush=True)
    else:
        backend = {"cpu": "cpu", "cuda": "cu132", "rocm": "rocm7.2"}[args.backend]
        if args.backend == "rocm" and not sys.platform.startswith("linux"):
            parser.error("ROCm PyTorch wheels are supported here only on Linux.")
        print(f"Using the requested {args.backend} PyTorch backend.", flush=True)

    commands = []
    if not (VENV / "bin/python").is_file():
        commands.append(["uv", "venv", "--python", ">=3.10", str(VENV)])
    commands.extend((
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(VENV / "bin/python"),
            "--torch-backend",
            backend,
            "--reinstall-package",
            "torch",
            "-e",
            ".",
        ],
        [str(VENV / "bin/python"), "-m", "ascii_optimizer.launcher", "-m", "ascii_optimizer.runtime", "--backend",
         "rocm" if backend.startswith("rocm") else "cuda" if backend != "cpu" else "cpu"],
    ))
    try:
        for command in commands:
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    except FileNotFoundError:
        raise SystemExit("uv is required. Install uv or use the pip fallback in README.md.") from None


if __name__ == "__main__":
    main()
