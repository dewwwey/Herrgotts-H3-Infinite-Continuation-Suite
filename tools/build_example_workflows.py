#!/usr/bin/env python3
"""Convert the split API workflows into loadable ComfyUI example workflows.

Reads ``workflows/job1_start_api.json`` and ``workflows/job2_continue_api.json``
and writes the frontend graph format into ``examples/``, using the live node
schemas for widget order and defaults so the widget lists are correct.

Usage: python tools/build_example_workflows.py [--base http://127.0.0.1:8188]
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflows"
EXAMPLES = ROOT / "examples"

REGISTRY_ID = "herrgotts-h3-infinite-continuation-suite"
RELEASE_VERSION = "1.4.0"
FRONTEND_VERSION = "1.53.10"

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

# Manual grid placement: (column, row). Keeps each example readable.
LAYOUT: dict[str, tuple[int, int]] = {
    "VAELoader": (0, 0),
    "CLIPLoader": (0, 1),
    "UNETLoader": (0, 2),
    "LoraLoaderModelOnly": (0, 3),
    "ImageDimensionsFromMegaPixels": (0, 4),
    "PrimitiveInt": (0, 5),
    "PatchSageAttentionKJ": (1, 0),
    "SolAttnPatch": (1, 1),
    "MiniMaxH3SigmaShift": (1, 2),
    "ModelPreviewOverrideKJ": (1, 3),
    "FantasticSeeds": (1, 4),
    "MiniMaxH3MediaLoader": (2, 0),
    "MiniMaxH3RefModStack": (2, 1),
    "MiniMaxH3PromptBuilder": (2, 2),
    "MiniMaxH3FantasticRefModTextEncode": (3, 0),
    "H3ContinuousStartExternalConditioning": (3, 1),
    "H3ContinuousContinueExternalConditioning": (3, 1),
    "H3ContinuousLoadLatent": (3, 2),
    "KSamplerSelect": (4, 0),
    "BasicScheduler": (4, 1),
    "BasicGuider": (4, 2),
    "RandomNoise": (4, 3),
    "SamplerCustomAdvanced": (5, 0),
    "VAEDecode": (6, 0),
    "VAEDecodeAudio": (6, 1),
    "H3ContinuousAnalyzeHandoverV14": (6, 2),
    "H3ContinuousStitchOutputV14": (7, 0),
    "H3ContinuousSaveLatent": (7, 1),
    "H3ContinuousStitchSavedChainV14": (3, 0),
    "PreviewAny": (7, 3),
    "CreateVideo": (8, 0),
    "SaveVideo": (8, 1),
    "SaveImage": (8, 2),
}

POS = {
    0: (0, 0),
    1: (520, 0),
    2: (1040, 0),
    3: (1560, 0),
    4: (2080, 0),
    5: (2600, 0),
    6: (3120, 0),
    7: (3640, 0),
    8: (4160, 0),
}
ROW_HEIGHT = 210
COLUMN_EXTRA = {3: 660, 7: 240}  # wider columns for the big conditioning nodes


def fetch_object_info(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/object_info", timeout=180) as response:
        return json.load(response)


def widget_specs(schema: dict) -> list[tuple[str, object, bool]]:
    """(name, default, is_widget) in widget order for a node schema."""
    out = []
    for section in ("required", "optional"):
        for name, spec in (schema.get("input", {}).get(section) or {}).items():
            if not isinstance(spec, list) or not spec:
                continue
            type_name = spec[0]
            options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(type_name, list):
                type_name = "COMBO"
            if type_name not in WIDGET_TYPES or options.get("forceInput"):
                continue
            if type_name == "COMBO":
                choices = spec[0] if isinstance(spec[0], list) else (options.get("options") or [])
                default = options.get("default", choices[0] if choices else None)
            else:
                default = options.get("default")
            out.append((name, default, True))
    return out


def widget_values(schema: dict, api_inputs: dict) -> list:
    values = []
    for name, default, _ in widget_specs(schema):
        if name in api_inputs and not isinstance(api_inputs[name], list):
            values.append(api_inputs[name])
        else:
            values.append(default)
    return values


def input_type(spec) -> str:
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return "COMBO"
        return spec[0] if isinstance(spec[0], str) else "*"
    return "*"


def build_graph(api: dict, object_info: dict, title_prefix: str) -> dict:
    nodes = []
    links = []
    link_id = 0

    # Stable ordering: input nodes first, then by (column, row).
    ordered = sorted(
        api.items(),
        key=lambda item: (
            LAYOUT.get(item[1]["class_type"], (9, 9)),
            item[0],
        ),
    )
    # API node ids may be non-numeric; map each to a stable frontend integer id.
    id_map: dict[str, int] = {}
    for position, (api_id, _node_data) in enumerate(ordered, start=1):
        id_map[api_id] = int(api_id) if api_id.isdigit() else 1000 + position

    for index, (nid, node) in enumerate(ordered):
        class_type = node["class_type"]
        schema = object_info.get(class_type, {"input": {}, "output": [], "output_name": []})
        declared = {}
        for section in ("required", "optional"):
            declared.update(schema.get("input", {}).get(section) or {})
        column, row = LAYOUT.get(class_type, (9, index))
        pos = POS.get(column, (0, 0))
        width = 620 + COLUMN_EXTRA.get(column, 0)
        height = 430

        inputs = []
        for name, spec in declared.items():
            if not isinstance(spec, list) or not spec:
                continue
            options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            type_name = input_type(spec)
            # A widget-typed input still becomes a socket when the graph actually
            # links into it (ComfyUI converts such sockets on load).
            is_widget = type_name in WIDGET_TYPES and not options.get("forceInput")
            if is_widget and not isinstance((node.get("inputs") or {}).get(name), list):
                continue
            slot = {"name": name, "type": type_name, "link": None}
            if section == "optional" or options.get("lazy") or name not in (schema.get("input", {}).get("required") or {}):
                slot["shape"] = 7
            inputs.append(slot)
        input_index = {slot["name"]: position for position, slot in enumerate(inputs)}

        outputs = []
        for position, out_type in enumerate(schema.get("output") or []):
            out_name = (schema.get("output_name") or [])[position] if position < len(schema.get("output_name") or []) else out_type
            outputs.append({"name": out_name, "type": out_type, "links": []})

        # Links into this node.
        for name, value in (node.get("inputs") or {}).items():
            if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str):
                continue
            if name not in input_index:
                continue
            link_id += 1
            origin, origin_slot = value
            inputs[input_index[name]]["link"] = link_id
            links.append([link_id, id_map[origin], origin_slot,
                          id_map[nid], input_index[name], inputs[input_index[name]]["type"]])

        nodes.append({
            "id": id_map[nid],
            "_api_id": nid,
            "type": class_type,
            "pos": [pos[0], pos[1] + row * ROW_HEIGHT],
            "size": [width, height],
            "flags": {},
            "order": index,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "title": (node.get("_meta") or {}).get("title") or class_type,
            "properties": {
                "Node name for S&R": class_type,
                "cnr_id": REGISTRY_ID,
                "ver": RELEASE_VERSION,
            },
            "widgets_values": widget_values(schema, node.get("inputs") or {}),
        })

    # Fill each output's link list from the link table.
    by_id = {node["id"]: node for node in nodes}
    for link in links:
        origin_node = by_id.get(link[1])
        if origin_node is not None and link[2] < len(origin_node["outputs"]):
            origin_node["outputs"][link[2]]["links"].append(link[0])
    for node in nodes:
        for output in node["outputs"]:
            if not output["links"]:
                output["links"] = None

    max_row = max((LAYOUT.get(n["type"], (9, 0))[1] for n in nodes), default=0)
    return {
        "id": None,
        "revision": 1,
        "last_node_id": max(node["id"] for node in nodes),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {
            "ds": {"scale": 0.4, "offset": [200, 300]},
            "frontendVersion": FRONTEND_VERSION,
            "title": title_prefix,
        },
        "version": 0.4,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8188")
    args = parser.parse_args()

    object_info = fetch_object_info(args.base.rstrip("/"))
    EXAMPLES.mkdir(exist_ok=True)

    jobs = [
        ("job1_start_api.json", "Herrgotts_H3_Infinite_v1.5_ExternalConditioning_01_Start.json",
         "H3 External Conditioning - Job 1: Start (segment 1)"),
        ("job2_continue_api.json", "Herrgotts_H3_Infinite_v1.5_ExternalConditioning_02_Continue.json",
         "H3 External Conditioning - Job 2: Continue (segment 2)"),
        ("job3_stitch_api.json", "Herrgotts_H3_Infinite_v1.5_ExternalConditioning_03_Stitch.json",
         "H3 External Conditioning - Job 3: Final Stitch"),
    ]
    for source, target, title in jobs:
        api = json.loads((WORKFLOW_DIR / source).read_text(encoding="utf-8"))
        graph = build_graph(api, object_info, title)
        path = EXAMPLES / target
        path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(graph['nodes'])} nodes, {len(graph['links'])} links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
