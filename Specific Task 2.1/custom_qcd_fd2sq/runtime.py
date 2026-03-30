"""Runtime dependency checks for the QCD custom graph encoder experiments."""

from __future__ import annotations

import importlib
import subprocess
import sys


def _clear_torch_geometric_modules() -> None:
    for name in list(sys.modules):
        if name == "torch_geometric" or name.startswith("torch_geometric."):
            sys.modules.pop(name, None)


def ensure_torch_geometric(auto_install: bool = False) -> None:
    try:
        module = importlib.import_module("torch_geometric")
        getattr(module, "typing")
        return
    except Exception as exc:
        if not auto_install:
            raise RuntimeError(
                "Failed to import `torch_geometric` cleanly. "
                "Install or repair PyG in this environment, e.g. "
                "`python -m pip install --upgrade --force-reinstall --no-cache-dir torch_geometric`."
            ) from exc

    _clear_torch_geometric_modules()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "torch_geometric",
        ],
        check=True,
    )
    _clear_torch_geometric_modules()

    module = importlib.import_module("torch_geometric")
    getattr(module, "typing")
