#!/usr/bin/env python3
"""Build the two-job external-conditioning workflows from the source workflow.

Input: the live "Fantastic Refmod" API-format workflow (10 s, two shots).
Output:
  * ``workflows/job1_start_api.json``    - 5 s segment 1 through H3 Start - External Conditioning
  * ``workflows/job2_continue_api.json`` - 5 s segment 2 through H3 Continue - External Conditioning,
                                           ending in the v1.4 saved-chain stitch

The split is intentionally explicit: each file is a complete, separately
submittable ComfyUI API prompt, and all state passes between them through the
saved AV latent + handover metadata on disk.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflows"

LATENT_PREFIX = "h3_external_conditioning/shot"
VIDEO_PREFIX = "video/H3_External_Conditioning"
# ---------------------------------------------------------------------------
# Node ids (stable, referenced by both jobs and by the docs)
# ---------------------------------------------------------------------------
VAE_VIDEO = "3"
VAE_AUDIO = "4"
BUILDER = "6"
MEDIA_LOADER = "7"
REFMOD_STACK = "9"
CLIP = "12"
UNET = "13"
PATCH_SAGE = "15"
PATCH_SOL = "16"
SIGMA_SHIFT = "17"
SAMPLER_SELECT = "18"
SCHEDULER = "19"
GUIDER = "20"
PREVIEW = "21"
SAMPLER = "22"
DECODE_AUDIO = "23"
DECODE_VIDEO = "24"
NOISE = "27"
LORA = "28"
CREATE_VIDEO = "31"
DIMENSIONS = "34"
SEED = "43"

# New nodes
EXTERNAL_ENCODE_1 = "11"       # replaces the in-graph text encode for job 1
LOAD_LATENT = "60"
ENCODE_2 = "61"
LOAD_LATENT_2 = "62"
ANALYZE_2 = "63"
SAVE_LATENT_2 = "64"
STITCH_OUTPUT_1 = "65"
SAVE_LATENT_1 = "66"
ANALYZE_1 = "67"
STITCH_OUTPUT_2 = "68"
STITCH_CHAIN = "69"
PREVIEW_REF_INFO_1 = "70"
PREVIEW_HANDOVER_2 = "71"
STITCH_OUTPUT_VIDEO = "72"
STITCH_SAVE_VIDEO = "73"
SAVE_LAST_FRAME_1 = "74"

# Job 1 writes its post-trim join frame here; job 2 loads it back through the
# Media Loader as <Picture 1>. SaveImage appends a frame counter on every run,
# so delete stale handover_last_frame_*.png before re-running job 1 (the
# generated name below is the one a fresh run produces).
LAST_FRAME_PREFIX = "h3_external_conditioning/handover_last_frame"
LAST_FRAME_FILE = "h3_external_conditioning/handover_last_frame_00001_.png"


def _node(class_type: str, title: str, inputs: dict, **extra) -> dict:
    node = {"inputs": inputs, "class_type": class_type, "_meta": {"title": title}}
    node.update(extra)
    return node


def _load_source(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [nid for nid in (BUILDER, REFMOD_STACK, CLIP, UNET, SAMPLER, SEED) if nid not in data]
    if missing:
        raise SystemExit(f"source workflow is missing expected node ids: {missing}")
    return data


def _generation_geometry(source: dict) -> tuple[int, int]:
    """Mirror the source graph's width/height so both jobs sample the same canvas."""
    dims = source.get(DIMENSIONS, {}).get("inputs", {})
    megapixels = float(dims.get("megapixels", 1.0))
    multiple = int(dims.get("multiple_of", 32))

    def compute(width: int, height: int) -> tuple[int, int]:
        ratio = width / height
        target = megapixels * 1_000_000
        new_h = (target / ratio) ** 0.5
        new_w = new_h * ratio
        return (int(round(new_w / multiple) * multiple), int(round(new_h / multiple) * multiple))

    picture = source.get(MEDIA_LOADER, {}).get("inputs", {}).get("media_state")
    try:
        media = json.loads(picture)[0]
        width, height = int(media["width"]), int(media["height"])
        mode = "from Media Loader media_state"
    except Exception:
        width, height, mode = 992, 992, "fallback 1:1 square"
    out = compute(width, height)
    print(f"generation geometry: {out[0]}x{out[1]} ({mode}: {width}x{height}, "
          f"{megapixels} MP, multiple_of {multiple})")
    return out


# ---------------------------------------------------------------------------
# Graph surgery
# ---------------------------------------------------------------------------

def _shared_nodes(source: dict) -> dict:
    """Model / sampler / decoder nodes shared by both jobs."""
    return {nid: copy.deepcopy(source[nid]) for nid in (
        VAE_VIDEO, VAE_AUDIO, BUILDER, MEDIA_LOADER, REFMOD_STACK, CLIP, UNET,
        PATCH_SAGE, PATCH_SOL, SIGMA_SHIFT, SAMPLER_SELECT, SCHEDULER, GUIDER,
        PREVIEW, SAMPLER, DECODE_AUDIO, DECODE_VIDEO, NOISE, LORA, SEED,
    ) if nid in source}


def _picture_wiring() -> dict:
    return {
        "prompt": [BUILDER, 0],
        "width": [DIMENSIONS, 0],
        "height": [DIMENSIONS, 1],
        "length": [SEGMENT_LENGTH_NODE, 0],
        "ref_image_size": "match",
        "reference_fps": 24,
        "max_total_tokens": 0,
        "voice_description_at_label": False,
        "stack_pictures": "every 4th",
        "clip": [CLIP, 0],
        "mods": [BUILDER, 20],
        "references": [BUILDER, 19],
        "vae": [VAE_VIDEO, 0],
        "audio_vae": [VAE_AUDIO, 0],
    }


# A 5 s segment on H3's 17k+5 grid: 5 * 24 = 120 requested -> 124 frames.
SEGMENT_FRAMES = 124
SEGMENT_DURATION = 5.0
# Frame count handed to Fantastic for its ordinary media-reference trimming. It
# does NOT decide the sampled segment length: the external-conditioning node's
# ``duration`` does. 124 = 5 s on H3's 17k+5 frame grid.
SEGMENT_LENGTH_NODE = "29"


def _dimensions_node(source: dict) -> dict:
    return _node("ImageDimensionsFromMegaPixels", "Dimensions From MP", {
        "megapixels": source.get(DIMENSIONS, {}).get("inputs", {}).get("megapixels", 0.5),
        "multiple_of": 32,
        "image": [BUILDER, 1],
    })


def build_job1(source: dict, width: int, height: int) -> dict:
    nodes = _shared_nodes(source)
    # Segment 1's own prompt: the truck in starts here and is still in motion at
    # the end of the shot, so segment 2 can continue it without a cut.
    nodes[BUILDER]["inputs"]["prompt_text"] = SEGMENT_1_PROMPT
    nodes[DIMENSIONS] = _dimensions_node(source)
    nodes[SEGMENT_LENGTH_NODE] = _node("PrimitiveInt", "Segment Frames (124 = 5 s)", {
        "value": SEGMENT_FRAMES,
    })

    # Segment 1: 5 s through the new external-conditioning Start node.
    nodes[EXTERNAL_ENCODE_1] = _node(
        "H3ContinuousStartExternalConditioning", "H3 Start \u2014 External Conditioning", {
            "conditioning": ["11e", 0],
            "width": [DIMENSIONS, 0],
            "height": [DIMENSIONS, 1],
            "duration": SEGMENT_DURATION,
            "reference_map": ["11e", 1],
        })
    # The Fantastic encoder still owns prompt/RefMod encoding, but its own latent
    # is deliberately unused: the new node builds the generation latent.
    nodes["11e"] = _node("MiniMaxH3FantasticRefModTextEncode", "Fantastic H3 RefMod Text Encode",
                         _picture_wiring())

    nodes[SAMPLER]["inputs"]["latent_image"] = [EXTERNAL_ENCODE_1, 1]

    nodes[STITCH_OUTPUT_1] = _node("H3ContinuousStitchOutputV14", "Output / Stitch \u2014 Masked AV v1.4", {
        "images": [DECODE_VIDEO, 0],
        "output_mode": "Stitch Ready",
        "handover": [ANALYZE_1, 0],
        "audio": [DECODE_AUDIO, 0],
        "head_context_frames": 0,
    })
    nodes[ANALYZE_1] = _node("H3ContinuousAnalyzeHandoverV14", "Auto Handover \u2014 Masked AV v1.4",
                             _analyze_inputs())
    nodes[SAVE_LATENT_1] = _node("H3ContinuousSaveLatent", "Save AV Latent", {
        "latent": [SAMPLER, 0],
        "filename_prefix": LATENT_PREFIX,
        "clip_index": 1,
        "handover": [ANALYZE_1, 0],
    })
    nodes[PREVIEW_REF_INFO_1] = _node("PreviewAny", "Start Info", {"source": [EXTERNAL_ENCODE_1, 2]})

    # Render this segment's visible body: Stitch Ready removes the unusable tail.
    nodes[CREATE_VIDEO] = _node("CreateVideo", "Create Video", {
        "fps": 24,
        "images": [STITCH_OUTPUT_1, 0],
        "audio": [STITCH_OUTPUT_1, 1],
    })
    nodes["44"] = _node("SaveVideo", "Save Video", {
        "filename_prefix": f"{VIDEO_PREFIX}_01", "format": "auto",
        "codec": "auto", "video": [CREATE_VIDEO, 0],
    })
    # Export the exact post-trim join frame so job 2's encoder can see the
    # previous scene: output 5 is the new last_frame IMAGE output.
    nodes[SAVE_LAST_FRAME_1] = _node("SaveImage", "Save Handover Last Frame", {
        "images": [STITCH_OUTPUT_1, 5],
        "filename_prefix": LAST_FRAME_PREFIX,
    })
    return nodes


def _analyze_inputs() -> dict:
    # Upstream v1.4 defaults (keep in sync with H3ContinuousAnalyzeHandoverV14).
    return {
        "images": [DECODE_VIDEO, 0],
        "preset": "Balanced",
        "analysis_window": 72,
        "freeze_hold": 8,
        "safety_margin": 3,
        "context_frames": "39",
        "analysis_size": 192,
        "final_mean_diff_threshold": 0.012,
        "final_active_pixel_threshold": 0.025,
        "max_final_active_area_percent": 3.0,
        "transition_mean_diff_threshold": 0.002,
        "transition_active_pixel_threshold": 0.010,
        "max_transition_active_area_percent": 1.0,
        "min_static_transition_percent": 70.0,
        "max_consecutive_motion_outliers": 2,
        "final_reference_frames": 15,
        "min_final_match_percent": 75.0,
        "max_consecutive_final_outliers": 3,
        "safety_mode": "fixed",
    }


def build_job2(source: dict, width: int, height: int) -> dict:
    nodes = _shared_nodes(source)
    # Job 2's own segment prompt replaces the builder's 10 s two-shot text.
    nodes[BUILDER]["inputs"]["prompt_text"] = SEGMENT_2_PROMPT
    nodes[BUILDER]["inputs"]["builder_state"] = SEGMENT_2_BUILDER_STATE
    # The Media Loader swaps the source's first-frame picture for job 1's saved
    # handover last frame: <Picture 1> becomes the exact frame [Shot 2] starts
    # from, which is what keeps the continuation in the same scene.
    nodes[MEDIA_LOADER]["inputs"]["media_state"] = json.dumps([_last_frame_media_item(width, height)])

    nodes[DIMENSIONS] = _dimensions_node(source)
    nodes[SEGMENT_LENGTH_NODE] = _node("PrimitiveInt", "Segment Frames (124 = 5 s)", {
        "value": SEGMENT_FRAMES,
    })
    nodes[LOAD_LATENT] = _node("H3ContinuousLoadLatent", "Load AV Latent", {
        "latent_path": LATENT_PREFIX, "clip_index": 1,
    })
    nodes[ENCODE_2] = _node(
        "H3ContinuousContinueExternalConditioning", "H3 Continue \u2014 External Conditioning", {
            "conditioning": ["11e2", 0],
            "previous_latent": [LOAD_LATENT, 0],
            "handover": [LOAD_LATENT, 3],
            "width": width,
            "height": height,
            "duration": SEGMENT_DURATION,
            "masked_context_frames": "39",
            "duration_mode": "Net New Content",
            "audio_tail_carryover": "Full Previous Tail",
            "audio_feather_ticks": 0,
            "reference_map": ["11e2", 1],
        })
    nodes["11e2"] = _node("MiniMaxH3FantasticRefModTextEncode", "Fantastic H3 RefMod Text Encode",
                          _picture_wiring())
    nodes[SAMPLER]["inputs"]["latent_image"] = [ENCODE_2, 1]
    # The guider consumes the external node's conditioning, not a stale encoder id.
    nodes[GUIDER]["inputs"]["conditioning"] = [ENCODE_2, 0]

    nodes[ANALYZE_2] = _node("H3ContinuousAnalyzeHandoverV14", "Auto Handover \u2014 Masked AV v1.4",
                             _analyze_inputs())
    nodes[SAVE_LATENT_2] = _node("H3ContinuousSaveLatent", "Save AV Latent", {
        "latent": [SAMPLER, 0],
        "filename_prefix": LATENT_PREFIX,
        "clip_index": 2,
        "handover": [ANALYZE_2, 0],
        "head_context_frames": [ENCODE_2, 2],
    })
    nodes[STITCH_OUTPUT_2] = _node("H3ContinuousStitchOutputV14", "Output / Stitch \u2014 Masked AV v1.4", {
        "images": [DECODE_VIDEO, 0],
        "output_mode": "Final Clip",
        "handover": [ANALYZE_2, 0],
        "audio": [DECODE_AUDIO, 0],
        "head_context_frames": [ENCODE_2, 2],
    })
    nodes[PREVIEW_HANDOVER_2] = _node("PreviewAny", "Continue Handover Info", {"source": [ENCODE_2, 4]})
    nodes["70b"] = _node("PreviewAny", "Continue Reference Info", {"source": [ENCODE_2, 5]})

    # Render this segment's visible body.
    nodes[CREATE_VIDEO] = _node("CreateVideo", "Create Video", {
        "fps": 24,
        "images": [STITCH_OUTPUT_2, 0],
        "audio": [STITCH_OUTPUT_2, 1],
    })
    nodes["44"] = _node("SaveVideo", "Save Video", {
        "filename_prefix": f"{VIDEO_PREFIX}_02", "format": "auto",
        "codec": "auto", "video": [CREATE_VIDEO, 0],
    })
    # The final stitch is job 3: Stitch Saved Chain reads the clip files straight
    # from disk and, with no data dependency on Save AV Latent, would otherwise
    # run before clip 2 exists and skip.
    return nodes


def build_job3(source: dict, width: int, height: int) -> dict:
    """Final stitch only: join the saved AV latents on disk."""
    nodes = {nid: copy.deepcopy(source[nid]) for nid in (VAE_VIDEO, VAE_AUDIO)}
    nodes[STITCH_CHAIN] = _node("H3ContinuousStitchSavedChainV14",
                                "Stitch Saved Chain \u2014 Masked AV v1.4", {
        "video_vae": [VAE_VIDEO, 0],
        "audio_vae": [VAE_AUDIO, 0],
        "latent_prefix": LATENT_PREFIX,
        "first_clip": 1,
        "last_clip": 2,
        "filename_prefix": f"{VIDEO_PREFIX}_final",
        "video_crossfade_frames": 4,
        "audio_crossfade_ms": 15.0,
        "luminance_match": False,
        "luminance_fade_frames": 16,
        "max_luminance_correction_percent": 10.0,
        "crf": 18,
        "max_safe_tail_bridge_frames": 0,
    })
    nodes["80"] = _node("PreviewAny", "Stitch Info", {"source": [STITCH_CHAIN, 1]})
    return nodes


def _last_frame_media_item(width: int, height: int) -> dict:
    """Media Loader entry for job 1's saved handover last frame (a picture)."""
    return {
        "kind": "picture",
        "file": f"{LAST_FRAME_FILE} [output]",
        "name": LAST_FRAME_FILE.rsplit("/", 1)[-1],
        "duration": None,
        "width": int(width),
        "height": int(height),
        "has_audio": False,
        "audio_mode": "off",
        "uid": "h3ec_handover_last_frame",
    }


SEGMENT_1_PROMPT = """subject_definitions:
<Subject 1> is a 21-year-old caucasian girl. Their name is Daniela.
<Picture 1> is the first frame of [Shot 1], showing <Subject 1> hunched over, talking to a little girl, <Subject 2>
<Video 1> is <Subject 1>
<Subject 2> is the little girl wrapped in a towel. Their name is Kelly.

summary:
[keyframe completion + reference generation] <Subject 2> Kelly hands an ice cream to <Subject 1> Daniela, who takes it and enjoys it while the camera starts a slow truck in.

retention_analysis:
<Picture 1> ([Shot 1] first frame): fully_preserved - it is used as the opening frame of [Shot 1] exactly as given.
<Subject 1>: fully_preserved - 

detailed_description:
The target video is a live-action, candid video, made outdoors on a professional camera.
[Shot 1] The camera starts on the same wide framing as <Picture 1>, holding static while a hunched over <Subject 1> takes an ice cream handed to her by <Subject 2>. She thanks the girl - (S1) says: <d>[English] Thank you!</d>, then stands up straight and starts eating the ice cream. As she starts eating, the camera begins a slow, steady truck in toward her, and the truck in is still in motion when the shot ends.
[Shot 2] No cut: the very same shot continues, and the camera keeps trucking in until it stops on a close up of <Subject 1> Daniela enjoying the ice cream.

overall_soundscape:
Outdoor ambience fitting the scene.

non_diegetic_music:
N/A"""

SEGMENT_2_PROMPT = """subject_definitions:
<Subject 1> is a 21-year-old caucasian girl. Their name is Daniela.
<Picture 1> is the exact last frame of [Shot 1], showing <Subject 1> Daniela mid truck in, eating the ice cream at the lakeside campsite.
<Video 1> is <Subject 1>

summary:
[continuation] No cut: the truck in continues without interruption until it stops on a close up of <Subject 1> Daniela enjoying the ice cream.

retention_analysis:
<Picture 1>: fully_preserved - [Shot 2] starts from this exact frame and every element of it stays in place while the camera moves closer.
<Subject 1>: fully_preserved - the same person continues from the previous segment.

detailed_description:
The target video is a live-action, candid video, made outdoors on a professional camera.
[Shot 2] The same uninterrupted shot continues from <Picture 1> with no cut: the camera keeps the truck in that was already in motion at the end of [Shot 1] and pushes in gradually until it comes to rest on a close up of <Subject 1> Daniela taking another bite of the ice cream and smiling. The lakeside campsite, the people in the background, the daylight, her bikini top and shorts all remain exactly as in <Picture 1> for the whole shot - the only change is the camera moving closer. There is no cut, no new scene and no framing jump.

overall_soundscape:
Outdoor ambience fitting the scene, continuing from the previous shot.

non_diegetic_music:
N/A"""

SEGMENT_2_BUILDER_STATE = json.dumps({
    "version": 1,
    "mode": "REF",
    "off": {},
    "duration": 5,
    "p2Shot": 1,
    "lastShot": 1,
    "imd": "",
    "soundscape": "",
    "music": "N/A",
    "ref": {
        "subjectDefs": [
            {"text": "<Subject 1> is a 21-year-old caucasian girl.", "name": "Daniela", "role": None},
            {"text": "<Picture 1> is the exact last frame of [Shot 1], showing <Subject 1> Daniela mid truck in, eating the ice cream at the lakeside campsite.", "role": None},
            {"text": "<Video 1> is <Subject 1>", "role": None},
        ],
        "summaryTypes": ["reference generation"],
        "summaryText": "!No cut: the truck in continues until it stops on a close up of Daniela enjoying the ice cream.",
        "retention": [
            {"label": "<Picture 1>", "context": "", "marker": "fully_preserved",
             "note": "[Shot 2] starts from this exact frame and every element of it stays in place while the camera moves closer."},
            {"label": "<Subject 1>", "context": "", "marker": "fully_preserved", "note": ""},
        ],
        "styleLine": "The target video is a live-action, candid video, made outdoors on a professional camera.",
        "detail": "[Shot 2] The same uninterrupted shot continues from <Picture 1> with no cut: the camera keeps the truck in that was already in motion at the end of [Shot 1] and pushes in gradually until it comes to rest on a close up of !Daniela taking another bite of the ice cream and smiling.",
        "soundscape": "Outdoor ambience fitting the scene, continuing from the previous shot.",
        "music": "N/A",
    },
}, separators=(",", ":"))


def to_api_format(nodes: dict) -> dict:
    """Drop UI-only keys and normalise the API prompt shape."""
    out = {}
    for nid, node in nodes.items():
        entry = {"inputs": node["inputs"], "class_type": node["class_type"]}
        if "_meta" in node:
            entry["_meta"] = node["_meta"]
        out[nid] = entry
    return out


def main() -> int:
    default_source = (Path("/home/dewey/.dsh/attachments/v1/files/fc")
                      / "fc688216c7157f813563d0d45f20bd3830354dcc0095eb926f9ab06f8b903e51"
                      / "Fantastic Refmod.json")
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_source
    source = _load_source(source_path)
    width, height = _generation_geometry(source)

    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    job1 = to_api_format(build_job1(source, width, height))
    job2 = to_api_format(build_job2(source, width, height))
    job3 = to_api_format(build_job3(source, width, height))
    for name, data in (("job1_start_api.json", job1),
                       ("job2_continue_api.json", job2),
                       ("job3_stitch_api.json", job3)):
        (WORKFLOW_DIR / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {WORKFLOW_DIR / name} ({len(data)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
