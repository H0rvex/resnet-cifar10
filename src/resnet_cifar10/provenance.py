"""Run and checkpoint provenance (git, torch, CUDA) for reproducible experiments."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def get_git_commit() -> str | None:
    """Return short SHA of HEAD, or None if not in a git repo or git unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def collect_run_provenance(device: torch.device) -> dict[str, Any]:
    """Lightweight metadata for checkpoints and run_info.json."""
    out: dict[str, Any] = {
        "torch_version": torch.__version__,
        "git_commit": get_git_commit(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        out["cuda_device_name"] = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        out["cuda_capability"] = f"{cap[0]}.{cap[1]}"
    return out


def write_run_info(path: str | Path, cfg_dict: dict[str, Any], provenance: dict[str, Any]) -> None:
    """Write human-readable JSON next to checkpoints (no torch.load needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": cfg_dict, "provenance": provenance}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
