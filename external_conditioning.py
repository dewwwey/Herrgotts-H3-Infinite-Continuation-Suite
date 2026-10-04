"""H3 Infinite - External Conditioning (Start / Continue).

Two nodes that build MiniMax H3 generation latents and temporal constraints from
**already encoded** H3 ``CONDITIONING``, without owning a text encoder.

Who owns what
-------------

* **Fantastic** (``MiniMaxH3FantasticRefModTextEncode``) owns the prompt, the
  RefMod bundle, the stored reference images and the ``reference_map``
  diagnostic. It returns ``(conditioning, reference_map, latent)``.
* **These nodes** own the generation latent: the empty H3 AV latent, the
  optional First/Last temporal anchors, and -- for Continue -- the v1.4 native
  Masked-AV protected context, its per-stream denoise masks and the safe
  handover slice.
* **Fantastic's ``latent`` output is unused here.** Its ``length`` does not
  decide the target latent; ``duration`` on these nodes does.

The execution path below contains **zero** tokenize / text-encode calls and
never imports Fantastic. The only contract shared with the encoder is the
standard H3 ``CONDITIONING`` metadata (``minimax_token_tags``, ``minimax_refs``,
``minimax_keyframes``).

Endpoint images are temporal constraints only
---------------------------------------------

First/Last Frame were historically also Qwen ``<Picture N>`` inputs. Here they
are **not**: they are VAE-encoded into the target latent and are never shown to
the text encoder. If a prompt needs the encoder to inspect an endpoint image,
present it through Fantastic before encoding and refer to it through Fantastic's
``reference_map``.

Fantastic's ordinary media inputs may also add *persistent DiT reference
blocks*, which ride through every sampling step. That is a different mechanism
from a temporal-only anchor.

Working state
-------------

Neither node reads or writes disk and neither keeps state between executions.
The previous segment's full sampled AV latent and handover metadata arrive on
the ``previous_latent`` / ``handover`` sockets, normally from **Load AV Latent**.
"""

import math

import torch

import nodes

from .latent_math import temporal_shape
from .nodes import (
    _empty_av_latent,
    _require_masked_av_support,
    _resize,
    _streams_from_latent,
    plan_masked_av_continuation,
)
from .release_utils import duration_to_requested_frames, masked_av_duration_plan

CATEGORY = "Herrgotts H3 Infinite Continuation Suite/External Conditioning"

MAX_DURATION_SECONDS = 150.0
MIN_DURATION_SECONDS = 2.0

_ANCHOR_TOOLTIP = (
    "Temporal anchor only. It is VAE-encoded into the target latent and is NOT presented to the "
    "text encoder, so it does not become <Picture N>. To have the encoder inspect an endpoint "
    "image, present it through Fantastic before encoding and address it via Fantastic's "
    "reference_map."
)
_VAE_TOOLTIP = (
    "MiniMax H3 video VAE. Needed only when a First/Last Frame anchor is connected; no VAE is "
    "required for a text/reference-only segment."
)
_REFERENCE_MAP_TOOLTIP = (
    "Diagnostic passthrough from Fantastic's RefMod Text Encode. Echoed back in the info outputs "
    "and never parsed to construct conditioning."
)
_CONDITIONING_TOOLTIP = (
    "Already encoded MiniMax H3 conditioning, normally Fantastic's RefMod Text Encode output. "
    "Must carry H3 minimax_token_tags. Do not apply a temporal-constraint node (First/Last Frame, "
    "Add Guide) before this socket: anchors are connected here instead."
)
_EXTERNAL_ENCODING_NOTE = (
    "Text/reference encoding: EXTERNAL (Fantastic MiniMaxH3FantasticRefModTextEncode). This node "
    "added no text-encode pass and no Qwen <Picture N> input; it only built the generation latent "
    "and its temporal constraints."
)

_ENTRY_STRUCTURE_ERROR = (
    "External conditioning entry {index} is malformed: expected a (embedding tensor, metadata "
    "dict) pair carrying H3 conditioning."
)
_TAGS_ERROR = (
    "Conditioning entry {index} has no compatible minimax_token_tags; use an H3 encoder "
    "(Fantastic RefMod Text Encode or MiniMax H3 Reference to Video)."
)
_EMPTY_ERROR = "External conditioning is empty; connect an H3 text-encode conditioning output."
_KEYFRAME_CONFLICT_ERROR = (
    "Incoming conditioning already has temporal keyframes; supply raw reference conditioning and "
    "connect anchors here."
)
_FRAME_COUNT_ERROR = "Incoming minimax_frame_count does not match the planned target frame count."
_VAE_REQUIRED_ERROR = "Connect the MiniMax H3 video VAE when First/Last Frame anchors are supplied."
_VIDEO_VAE_ERROR = (
    "The VAE connected for the First/Last Frame anchor is not the MiniMax H3 video VAE. Audio VAEs "
    "and other model VAEs cannot encode H3 temporal anchors."
)


# ---------------------------------------------------------------------------
# Conditioning contract
# ---------------------------------------------------------------------------

def _format_reference_info(reference_map, anchor_count):
    """Report that text/reference encoding is external and echo Fantastic's map."""
    lines = [_EXTERNAL_ENCODING_NOTE]
    if anchor_count:
        lines.append(
            f"Temporal-only anchors attached: {anchor_count} (VAE-encoded into the latent; not "
            "presented to the text encoder)."
        )
    else:
        lines.append("Temporal-only anchors attached: none.")
    text = (reference_map or "").strip()
    if text:
        lines.append("Fantastic reference map:")
        lines.append(text)
    else:
        lines.append("Fantastic reference map: not supplied (reference_map input is empty).")
    return "\n".join(lines)


def _validate_external_conditioning(conditioning):
    """Validate already-encoded H3 conditioning; return immutable-safe entries.

    Every entry is checked and its metadata copied. The caller's list, metadata
    dicts and embedding tensors are never mutated, and embedding tensors are
    passed through by identity (they are large and are only read downstream).
    """
    if conditioning is None:
        raise ValueError(_EMPTY_ERROR)
    try:
        entries = list(conditioning)
    except TypeError as exc:
        raise ValueError(_EMPTY_ERROR) from exc
    if not entries:
        raise ValueError(_EMPTY_ERROR)

    validated = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(_ENTRY_STRUCTURE_ERROR.format(index=index))
        embedding, metadata = entry[0], entry[1]
        if not torch.is_tensor(embedding):
            raise ValueError(_ENTRY_STRUCTURE_ERROR.format(index=index))
        if embedding.ndim < 2:
            raise ValueError(
                f"Conditioning entry {index} embedding must be a batched sequence tensor, got "
                f"shape {tuple(embedding.shape)}."
            )
        if not isinstance(metadata, dict):
            raise ValueError(_ENTRY_STRUCTURE_ERROR.format(index=index))

        tags = metadata.get("minimax_token_tags")
        if not torch.is_tensor(tags) or tags.ndim != 1:
            raise ValueError(_TAGS_ERROR.format(index=index))
        if int(tags.shape[0]) != int(embedding.shape[1]):
            raise ValueError(
                f"Conditioning entry {index} has {int(tags.shape[0])} minimax_token_tags for an "
                f"embedding sequence of {int(embedding.shape[1])} positions; the conditioning is "
                "inconsistent or does not come from this H3 encoder."
            )

        if metadata.get("minimax_keyframes"):
            raise ValueError(_KEYFRAME_CONFLICT_ERROR)

        validated.append((embedding, dict(metadata)))
    return validated


def _attach_constraints(entries, frame_count, keyframes):
    """Attach the new temporal constraints to pre-validated conditioning entries.

    Every entry keeps its own embedding, device, dtype, token tags, schedule
    bounds, unknown metadata and its own reference list (copied, never mutated).
    """
    for index, (_, metadata) in enumerate(entries):
        incoming = metadata.get("minimax_frame_count")
        if incoming is not None and int(incoming) != int(frame_count):
            raise ValueError(
                f"{_FRAME_COUNT_ERROR} Entry {index} carries {int(incoming)}, planned {int(frame_count)}."
            )

    out = []
    for embedding, metadata in entries:
        new_metadata = dict(metadata)
        new_metadata["minimax_keyframes"] = keyframes if keyframes else []
        refs = metadata.get("minimax_refs")
        if refs:
            new_metadata["minimax_refs"] = [dict(ref) for ref in refs]
        elif "minimax_refs" in metadata:
            new_metadata["minimax_refs"] = []
        # The native H3 layout does not read a frame count from conditioning, so
        # only carry one forward when the caller already had one.
        if "minimax_frame_count" in metadata:
            new_metadata["minimax_frame_count"] = int(frame_count)
        out.append([embedding, new_metadata])
    return out


# ---------------------------------------------------------------------------
# Geometry, duration and temporal anchors
# ---------------------------------------------------------------------------

def _validate_geometry(width, height):
    width = int(width)
    height = int(height)
    limit = int(nodes.MAX_RESOLUTION)
    for name, value in (("width", width), ("height", height)):
        if value < 32 or value > limit:
            raise ValueError(f"{name} must be between 32 and {limit}, got {value}")
        if value % 32 != 0:
            raise ValueError(f"{name} must be a multiple of 32 for MiniMax H3, got {value}")
    return width, height


def _validate_duration(duration):
    seconds = float(duration)
    if not math.isfinite(seconds):
        raise ValueError("duration must be a finite number of seconds")
    if seconds < MIN_DURATION_SECONDS or seconds > MAX_DURATION_SECONDS:
        raise ValueError(
            f"duration must be between {MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS} seconds, "
            f"got {seconds}"
        )
    return seconds


def _target_plan(duration):
    """Resolve the legal H3 frame count for a Start segment of ``duration`` seconds."""
    requested_frames = duration_to_requested_frames(duration)
    frame_count, _, _ = temporal_shape(requested_frames)
    return requested_frames, int(frame_count)


def _require_video_vae(vae):
    if vae is None:
        raise ValueError(_VAE_REQUIRED_ERROR)
    first_stage = getattr(vae, "first_stage_model", None)
    if first_stage is None:
        # A wrapper whose model is not loaded yet: let vae.encode() report the failure.
        return vae
    try:
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    except Exception:  # pragma: no cover - a core without native H3 support is rejected earlier
        return vae
    if not isinstance(first_stage, MiniMaxH3VideoVAE):
        raise ValueError(_VIDEO_VAE_ERROR)
    return vae


def _encode_anchor(vae, image, width, height, role):
    """Resize one endpoint image to the generation canvas and VAE-encode it once."""
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(
            f"{role} must be a ComfyUI IMAGE batch shaped [batch, height, width, channels >= 3]"
        )
    frames = image[:1]
    if int(frames.shape[1]) < 1 or int(frames.shape[2]) < 1:
        raise ValueError(f"{role} is empty")
    resized = _resize(frames, width, height, "disabled")
    latent = vae.encode(resized)
    if not torch.is_tensor(latent) or latent.ndim < 4:
        raise ValueError(
            f"The connected VAE returned an unusable {role} latent; connect the MiniMax H3 video VAE."
        )
    return latent


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class H3ContinuousStartExternalConditioning:
    """Segment 1 from externally encoded conditioning: T2VA / I2VA / L2VA / FL2VA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": _CONDITIONING_TOOLTIP}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "tooltip": "Generation width. Use the width Fantastic encoded its references at."}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "tooltip": "Generation height. Use the height Fantastic encoded its references at."}),
                "duration": ("FLOAT", {"default": 5.0, "min": MIN_DURATION_SECONDS,
                    "max": MAX_DURATION_SECONDS, "step": 0.1,
                    "tooltip": "Total duration of this first segment, snapped up to H3's 17k+5 frame grid "
                               "at 24 fps (5.0 s -> 124 frames). This is the segment length; the length "
                               "Fantastic used to encode references does not decide it."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {
                    "tooltip": "Optional First Frame temporal anchor (I2VA/FL2VA). " + _ANCHOR_TOOLTIP}),
                "last_frame": ("IMAGE", {
                    "tooltip": "Optional Last Frame temporal anchor (L2VA/FL2VA). " + _ANCHOR_TOOLTIP}),
                "vae": ("VAE", {"tooltip": _VAE_TOOLTIP}),
                "reference_map": ("STRING", {"forceInput": True, "tooltip": _REFERENCE_MAP_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "info")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Segment 1 (start) from already encoded H3 conditioning. Builds the H3 AV latent and "
        "optional First/Last temporal anchors. No CLIP, no prompt, no tokenization: Fantastic's "
        "RefMod Text Encode owns prompt and reference encoding. Connect neither, first, last or "
        "both anchor images for T2VA / I2VA / L2VA / FL2VA."
    )

    def build(self, conditioning, width, height, duration,
              first_frame=None, last_frame=None, vae=None, reference_map=None):
        width, height = _validate_geometry(width, height)
        seconds = _validate_duration(duration)
        entries = _validate_external_conditioning(conditioning)
        requested_frames, frame_count = _target_plan(seconds)

        keyframes = []
        if first_frame is not None or last_frame is not None:
            vae = _require_video_vae(vae)
        if first_frame is not None:
            keyframes.append({
                "resolved_frame_index": 0,
                "latent": _encode_anchor(vae, first_frame, width, height, "First Frame"),
            })
        if last_frame is not None:
            keyframes.append({
                "resolved_frame_index": frame_count - 1,
                "latent": _encode_anchor(vae, last_frame, width, height, "Last Frame"),
            })

        target_latent, latent_frame_count = _empty_av_latent(width, height, requested_frames)
        if int(latent_frame_count) != frame_count:
            raise RuntimeError(
                f"internal H3 plan mismatch: {latent_frame_count} latent frames vs {frame_count} planned"
            )

        positive = _attach_constraints(entries, frame_count, keyframes)

        first_present = first_frame is not None
        last_present = last_frame is not None
        if first_present and last_present:
            mode = "FL2VA"
        elif first_present:
            mode = "I2VA"
        elif last_present:
            mode = "L2VA"
        else:
            mode = "T2VA"
        info = (
            f"start=external_conditioning | mode={mode} | geometry {width}x{height} | "
            f"duration {seconds:.3f}s -> {requested_frames} requested frames -> {frame_count} H3 frames "
            f"({frame_count / 24.0:.3f}s) | temporal anchors {len(keyframes)} "
            f"(first={'yes' if first_present else 'no'}, last={'yes' if last_present else 'no'}) | "
            f"conditioning entries {len(entries)} | inherited refs "
            f"{sum(len(metadata.get('minimax_refs') or []) for _, metadata in entries)} (unchanged) | "
            f"text encoder passes added 0 | info only, not a Picture Map"
        )
        return (positive, target_latent, info)


class H3ContinuousContinueExternalConditioning:
    """Segment 2+ from externally encoded conditioning plus saved v1.4 Masked-AV state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": _CONDITIONING_TOOLTIP}),
                "previous_latent": ("LATENT", {
                    "tooltip": "Previous segment's FULL sampled AV latent (Load AV Latent), not trimmed "
                               "render frames and not an empty latent."}),
                "handover": ("H3_CONTINUOUS_HANDOVER", {
                    "tooltip": "Previous segment's v1.4 safe-handover metadata (Load AV Latent -> "
                               "handover). The protected video window ends BEFORE the unusable "
                               "freeze/brightness tail."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "tooltip": "Must match the previous video latent grid."}),
                "height": ("INT", {"default": 768, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32,
                    "tooltip": "Must match the previous video latent grid."}),
                "duration": ("FLOAT", {"default": 5.0, "min": MIN_DURATION_SECONDS,
                    "max": MAX_DURATION_SECONDS, "step": 0.1,
                    "tooltip": "Interpreted by Duration Mode. Net New Content targets this much visible "
                               "newly generated body after the protected head; Total Generation keeps "
                               "the whole-clip meaning."}),
                "masked_context_frames": (["39", "90", "141", "192"], {
                    "default": "39",
                    "tooltip": "Protected previous VIDEO context ending at the freeze-safe Auto Handover "
                               "boundary. Exact joint H3 video boundaries are 39/90/141/192 frames; 39 "
                               "(~1.625 s) is recommended."}),
                "duration_mode": (["Net New Content", "Total Generation"], {
                    "default": "Net New Content",
                    "tooltip": "Recommended: Net New Content. Picks the nearest legal H3 17k+5 total so "
                               "the visible new video body stays close to Duration."}),
                "audio_tail_carryover": (["Full Previous Tail", "Match Video Handover"], {
                    "default": "Full Previous Tail",
                    "tooltip": "Recommended for dialogue: keep the freeze-safe video handover, but "
                               "protect original audio through the previous segment's actual end so "
                               "word/phoneme endings survive. Match Video Handover reproduces the "
                               "video-matched behavior."}),
                "audio_feather_ticks": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1, "advanced": True,
                    "tooltip": "Experimental/legacy. Keep 0 for hard audio protection; nonzero values "
                               "let the final protected audio ticks be partially denoised and can "
                               "weaken phoneme continuity."}),
            },
            "optional": {
                # Appended after the upstream widget order so positional values from
                # saved workflows keep their meaning.
                "last_frame": ("IMAGE", {
                    "tooltip": "Optional future endpoint temporal anchor. " + _ANCHOR_TOOLTIP}),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE; required only when Last Frame is supplied. The "
                               "protected previous context supplies this segment's starting state, so "
                               "there is no First Frame input."}),
                "reference_map": ("STRING", {"forceInput": True, "tooltip": _REFERENCE_MAP_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "positive", "latent", "actual_head_context_frames", "ignored_tail_frames",
        "handover_info", "reference_info",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Segment 2+ from already encoded H3 conditioning plus the previous segment's full sampled AV "
        "latent and safe handover metadata. Reuses the v1.4 native Masked-AV planner unchanged: the "
        "protected previous video/audio context and its per-stream denoise masks live directly in the "
        "target latent, not in minimax_refs. No CLIP, no prompt, no tokenization."
    )

    def build(self, conditioning, previous_latent, handover, width, height, duration,
              masked_context_frames="39", duration_mode="Net New Content",
              audio_tail_carryover="Full Previous Tail", audio_feather_ticks=0,
              last_frame=None, vae=None, reference_map=None):
        width, height = _validate_geometry(width, height)
        seconds = _validate_duration(duration)
        entries = _validate_external_conditioning(conditioning)
        comfy_version = _require_masked_av_support()
        # Deliberately no _require_patches() here. On a current ComfyUI the
        # continuation constraints this node builds are all native: anchors use
        # the stock keyframe layout, and the protected AV context lives in
        # latent + noise_mask. The suite's legacy layout/audio-timeline wrapper is
        # only needed by the older marker-based paths, and requesting it would
        # make this node collide with unrelated H3 packs that wrap PackedLayout
        # (for example SolAttn). Demand that the caller install it separately if
        # they want the legacy behaviour.

        # Resolve the plan and fail early on a geometry mismatch, before any VAE work.
        context_request = int(masked_context_frames)
        duration_plan = masked_av_duration_plan(seconds, context_request, duration_mode)
        frame_count = int(duration_plan["total_frame_count"])
        prev_video, _prev_audio = _streams_from_latent(previous_latent)
        target_grid = (height // 16, width // 16)
        if tuple(prev_video.shape[-2:]) != target_grid:
            raise ValueError(
                "Previous latent resolution differs from the requested target resolution. "
                f"Previous video latent grid {tuple(prev_video.shape[-2:])}, target grid {target_grid}."
            )

        keyframes = []
        if last_frame is not None:
            vae = _require_video_vae(vae)
            keyframes.append({
                "resolved_frame_index": frame_count - 1,
                "latent": _encode_anchor(vae, last_frame, width, height, "Last Frame"),
            })

        target_latent, _ = _empty_av_latent(width, height, frame_count)
        target_latent, actual_context_frames, ignored_tail, handover_info, details = plan_masked_av_continuation(
            previous_latent, handover, target_latent, width, height, seconds,
            masked_context_frames=masked_context_frames,
            duration_mode=duration_mode,
            audio_tail_carryover=audio_tail_carryover,
            audio_feather_ticks=audio_feather_ticks,
        )

        positive = _attach_constraints(entries, frame_count, keyframes)

        audio_plan = details["audio_plan"]
        handover_info = (
            f"{handover_info} | ComfyUI {comfy_version} | external_conditioning_segment | "
            f"duration requested {seconds:.3f}s -> plan {duration_plan['duration_mode']} total "
            f"{frame_count}f/{duration_plan['total_seconds']:.3f}s | protected head "
            f"{actual_context_frames}f | discarded previous tail {ignored_tail}f | audio "
            f"{audio_plan['mode']} latent {details['audio_start_tick']}:{details['audio_end_tick']} "
            f"({details['audio_steps']} ticks) | temporal anchors {len(keyframes)} | conditioning "
            f"entries {len(entries)} | inherited refs "
            f"{sum(len(metadata.get('minimax_refs') or []) for _, metadata in entries)} (unchanged) | "
            f"text encoder passes added 0"
        )
        reference_info = _format_reference_info(reference_map, len(keyframes))
        return (positive, target_latent, actual_context_frames, ignored_tail,
                handover_info, reference_info)


NODE_CLASS_MAPPINGS = {
    "H3ContinuousStartExternalConditioning": H3ContinuousStartExternalConditioning,
    "H3ContinuousContinueExternalConditioning": H3ContinuousContinueExternalConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuousStartExternalConditioning": "H3 Start \u2014 External Conditioning",
    "H3ContinuousContinueExternalConditioning": "H3 Continue \u2014 External Conditioning",
}
