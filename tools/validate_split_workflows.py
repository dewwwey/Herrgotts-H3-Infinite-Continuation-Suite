#!/usr/bin/env python3
"""Validate the split workflows against a live ComfyUI instance.

Checks every node's class_type exists, every declared input is accepted by that
node's real schema, every link resolves to an existing node/output of the right
type, and required inputs are present. This is the fast pre-flight before
submitting real jobs.

The pack passed to ``--pack`` (default: this repository) is imported in-process
so its node schemas are known even when the running ComfyUI has not reloaded it.

Usage: python tools/validate_split_workflows.py [--base URL] [--pack ROOT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflows"

# Frontend-only pseudo-widgets ("Edit prompt…", "+ Media loader", …) never reach
# the backend and are not part of the node schema.
def _is_ui_artifact(name: str) -> bool:
    return not name.isidentifier()


def _is_opaque_type(spec) -> bool:
    """Autogrow / dynamic-combo inputs: the value shape is node-defined."""
    return isinstance(spec, list) and bool(spec) and isinstance(spec[0], str) and spec[0].startswith("COMFY_")


def load_pack_schemas(pack_root: Path) -> dict:
    """Build object_info-shaped schemas from a pack checkout, without ComfyUI loading it."""
    comfy_root = Path("/opt/comfyui/ComfyUI")
    if (comfy_root / "comfy").is_dir() and str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    name = "h3ext_schema_probe"
    spec = importlib.util.spec_from_file_location(
        name, pack_root / "__init__.py", submodule_search_locations=[str(pack_root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    schemas = {}
    for node_id, cls in module.NODE_CLASS_MAPPINGS.items():
        inputs = cls.INPUT_TYPES()
        schemas[node_id] = {
            "input": {
                "required": inputs.get("required", {}),
                "optional": inputs.get("optional", {}),
            },
            "output": list(getattr(cls, "RETURN_TYPES", ()) or ()),
            "output_name": list(getattr(cls, "RETURN_NAMES", ()) or ()),
        }
    return schemas


def fetch_object_info(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/object_info", timeout=180) as response:
        return json.load(response)


def schema_inputs(schema: dict) -> tuple[dict, set]:
    declared: dict = {}
    for section in ("required", "optional"):
        for name, spec in (schema.get("input", {}).get(section) or {}).items():
            declared[name] = spec
    required = set(schema.get("input", {}).get("required", {}) or {})
    return declared, required


def is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def validate(path: Path, object_info: dict) -> list[str]:
    errors: list[str] = []
    workflow = json.loads(path.read_text(encoding="utf-8"))

    for nid, node in workflow.items():
        class_type = node.get("class_type")
        schema = object_info.get(class_type)
        if schema is None:
            errors.append(f"{path.name}: node {nid} uses class_type {class_type!r}, unknown to this ComfyUI")
            continue
        declared, required = schema_inputs(schema)
        outputs = schema.get("output") or []
        given = node.get("inputs") or {}

        for name in sorted(required):
            if _is_ui_artifact(name):
                continue
            if name not in given:
                errors.append(f"{path.name}: node {nid} ({class_type}) missing required input {name!r}")
        for name, value in given.items():
            if _is_ui_artifact(name):
                continue
            if name not in declared:
                errors.append(f"{path.name}: node {nid} ({class_type}) has unknown input {name!r}")
                continue
            if is_link(value):
                target, index = value[0], value[1]
                if target not in workflow:
                    errors.append(f"{path.name}: node {nid}.{name} links to missing node {target!r}")
                    continue
                target_type = workflow[target].get("class_type")
                target_schema = object_info.get(target_type, {})
                target_outputs = target_schema.get("output") or []
                if not isinstance(index, int) or index < 0 or index >= len(target_outputs):
                    errors.append(
                        f"{path.name}: node {nid}.{name} links to {target} output {index}, but "
                        f"{target_type} has {len(target_outputs)} outputs"
                    )
                    continue
                expected = declared[name]
                if isinstance(expected, list) and expected and isinstance(expected[0], str):
                    wanted = expected[0]
                    got = target_outputs[index]
                    if got != wanted and wanted != "*" and got != "*":
                        errors.append(
                            f"{path.name}: node {nid}.{name} expects {wanted}, but {target}[{index}] "
                            f"({target_type}) provides {got}"
                        )
            else:
                spec = declared[name]
                if _is_opaque_type(spec):
                    continue
                if isinstance(spec, list) and spec and isinstance(spec[0], list):
                    if value not in spec[0]:
                        errors.append(
                            f"{path.name}: node {nid}.{name} value {value!r} is not one of {spec[0]}"
                        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8188")
    parser.add_argument("--pack", default=str(ROOT))
    parser.add_argument("--no-pack", action="store_true",
                        help="Only use the live ComfyUI schemas (no in-process pack import).")
    args = parser.parse_args()

    object_info = fetch_object_info(args.base.rstrip("/"))
    pack_schemas = {} if args.no_pack else load_pack_schemas(Path(args.pack))
    overridden = sorted(set(pack_schemas) - set(object_info))
    object_info.update(pack_schemas)
    if overridden:
        print(f"using in-process schema for not-yet-reloaded nodes: {', '.join(overridden)}")

    paths = sorted(WORKFLOW_DIR.glob("job*_api.json"))
    if not paths:
        raise SystemExit(f"no workflows found in {WORKFLOW_DIR}")

    all_errors: list[str] = []
    for path in paths:
        errors = validate(path, object_info)
        all_errors.extend(errors)
        print(f"{path.name}: {'OK' if not errors else f'{len(errors)} error(s)'}")

    if all_errors:
        print()
        for error in all_errors:
            print(" -", error)
        return 1
    print("all workflows match the live node schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
