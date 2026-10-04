"""Shared test bootstrap.

The external-conditioning tests import the real pack, which imports ComfyUI
(``comfy.model_management``, ``folder_paths``, ``node_helpers``, ...). Make the
ComfyUI checkout importable without requiring the caller to set PYTHONPATH.

Set ``COMFYUI_ROOT`` to point at a different ComfyUI checkout.
"""

import os
import sys
from pathlib import Path

DEFAULT_COMFYUI_ROOT = "/opt/comfyui/ComfyUI"


def _comfyui_candidates():
    env = os.environ.get("COMFYUI_ROOT")
    if env:
        yield Path(env)
    yield Path(DEFAULT_COMFYUI_ROOT)
    # Running from a source checkout next to the pack.
    yield Path(__file__).resolve().parents[2].parent / "ComfyUI"


for candidate in _comfyui_candidates():
    if (candidate / "comfy").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
