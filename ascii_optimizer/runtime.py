"""Select and verify the numerical device used by the renderer."""

from __future__ import annotations

import argparse

import torch


def select_device(requested: str = "auto") -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("GPU solving is unavailable. Run scripts/setup_environment.py "
                             "with --backend rocm or --backend cuda for your GPU.")
        backend = "ROCm" if torch.version.hip else "CUDA"
        print(f"Solver device: {torch.cuda.get_device_name(device)} ({backend}, PyTorch {torch.__version__})",
              flush=True)
    else:
        print(f"Solver device: {device} (PyTorch {torch.__version__})", flush=True)
        if device.type == "cpu" and not (torch.version.hip or torch.version.cuda):
            print("CPU-only PyTorch installed. For GPU solving, run scripts/setup_environment.py.", flush=True)
    return str(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm"), required=True)
    args = parser.parse_args()
    if args.backend == "rocm" and not torch.version.hip:
        raise SystemExit("Expected ROCm PyTorch, but the installed build has no HIP support.")
    if args.backend == "cuda" and not torch.version.cuda:
        raise SystemExit("Expected CUDA PyTorch, but the installed build has no CUDA support.")
    device = select_device("cpu" if args.backend == "cpu" else "cuda")
    # Discovery alone does not verify that the wheel supports this GPU's ISA.
    value = torch.arange(16., device=device, requires_grad=True)
    value.square().sum().backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    assert value.grad[-1].item() == 30
    print("Solver computation and backpropagation verified.", flush=True)


if __name__ == "__main__":
    main()
