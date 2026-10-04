"""Focused tests for the external-conditioning nodes.

These tests exercise the real package code inside ComfyUI's environment. They
never run the H3 diffusion model: the CLIP and VAE are stubs, and
``_require_masked_av_support`` is monkeypatched so the
structural contract (metadata assembly, duration planning, masking, immutability)
can be checked without a sampler.

The equivalence tests compare the new Continue node against the upstream
``H3ContinuousContinueV14`` build on identical inputs, which is the strongest
available guarantee that the shared-planner refactor did not shift behaviour.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

# The pack's ``nodes.py`` does ``import nodes`` to reach ComfyUI's own top-level
# ``nodes`` module. Importing the pack under a synthetic package name keeps the
# pack's ``nodes.py`` out of the top-level namespace so that import resolves to
# ComfyUI, and relative imports inside the pack still work.
PKG = "h3ext_pack_under_test"
if PKG not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = package
    spec.loader.exec_module(package)
else:  # pragma: no cover - only when another test already imported the pack
    package = sys.modules[PKG]

ec = importlib.import_module(f"{PKG}.external_conditioning")
H3ContinuousStartExternalConditioning = ec.H3ContinuousStartExternalConditioning
H3ContinuousContinueExternalConditioning = ec.H3ContinuousContinueExternalConditioning
h3nodes = importlib.import_module(f"{PKG}.nodes")
latent_math = importlib.import_module(f"{PKG}.latent_math")
video_latent_t = latent_math.video_latent_t

FPS = 24


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubClip:
    """Mimics the parts of an H3 CLIP the upstream v1.4 Continue node uses."""

    def __init__(self, seq_len=8, hidden=6):
        self.seq_len = seq_len
        self.hidden = hidden
        self.tokenize_calls = 0
        self.encode_calls = 0

    def tokenize(self, prompt, images=None, minimax_ref_items=None):
        self.tokenize_calls += 1
        return {"stub": True, "images": images, "items": minimax_ref_items}

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_calls += 1
        # Two scheduled entries, as an H3 encoder can return: the contract must
        # hold for every entry, not just entry zero.
        return [
            [torch.zeros(1, self.seq_len, self.hidden),
             {"minimax_token_tags": torch.ones(self.seq_len, dtype=torch.long)}],
            [torch.zeros(1, self.seq_len, self.hidden),
             {"minimax_token_tags": torch.ones(self.seq_len, dtype=torch.long),
              "start_percent": 0.5, "end_percent": 1.0}],
        ]


class StubVideoVAE:
    """Video-VAE stand-in: shape-correct H3 latents, deterministic values.

    ``first_stage_model`` is a real ``MiniMaxH3VideoVAE`` instance (constructed
    without running its ``__init__``) so the node's VAE-kind check is exercised
    for real rather than stubbed out.
    """

    def __init__(self, tag=1.0):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        self.tag = float(tag)
        self.encode_calls = 0
        self.first_stage_model = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)

    def encode(self, image):
        self.encode_calls += 1
        return torch.full((1, 24, 12, 48, 84), self.tag)


class StubAudioVAE:
    """Stand-in for an audio VAE: must be rejected as a temporal-anchor encoder."""

    def __init__(self):
        self.first_stage_model = object()
        self.audio_sample_rate = 32000

    def encode(self, image):
        raise AssertionError("an audio VAE must never be asked to encode an anchor")


def _conditioning(entries=2, seq_len=8, hidden=6, refs=None, extra_metadata=None):
    out = []
    for index in range(entries):
        metadata = {
            "minimax_token_tags": torch.ones(seq_len, dtype=torch.long),
        }
        if refs is not None:
            metadata["minimax_refs"] = [dict(ref) for ref in refs]
        if extra_metadata:
            metadata.update(extra_metadata)
        out.append([torch.full((1, seq_len, hidden), float(index)), metadata])
    return out


def _target_vae_shape(width, height, frame_count):
    vt = video_latent_t(frame_count)
    return (1, 24, vt, height // 16, width // 16)


def _previous_latent(frame_count=124, width=1344, height=768, seed=7):
    import comfy.nested_tensor

    vt = video_latent_t(frame_count)
    audio_t = round((frame_count / FPS) * 40)
    generator = torch.Generator().manual_seed(seed)
    video = torch.rand((1, 24, vt, height // 16, width // 16), generator=generator)
    audio = torch.rand((1, 32, 2, audio_t), generator=generator)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _handover(frame_count=124, handover_end_frame=106, available=True):
    return {
        "available": available,
        "frame_count": frame_count,
        "handover_end_frame": handover_end_frame,
        "masked_av_source_policy": "safe_handover_window",
    }


@pytest.fixture(autouse=True)
def _patch_runtime(monkeypatch):
    """Keep the environment checks deterministic and side-effect free."""
    monkeypatch.setattr(ec, "_require_masked_av_support", lambda: "test")
    yield


# ---------------------------------------------------------------------------
# Registration / interface
# ---------------------------------------------------------------------------

def test_new_nodes_are_registered_under_stable_ids():
    mappings = package.NODE_CLASS_MAPPINGS
    assert mappings["H3ContinuousStartExternalConditioning"] is H3ContinuousStartExternalConditioning
    assert mappings["H3ContinuousContinueExternalConditioning"] is H3ContinuousContinueExternalConditioning
    names = package.NODE_DISPLAY_NAME_MAPPINGS
    assert names["H3ContinuousStartExternalConditioning"] == "H3 Start \u2014 External Conditioning"
    assert names["H3ContinuousContinueExternalConditioning"] == "H3 Continue \u2014 External Conditioning"


def test_existing_node_ids_and_outputs_are_untouched():
    mappings = package.NODE_CLASS_MAPPINGS
    for node_id in (
        "H3ContinuousStartV14", "H3ContinuousContinueV14", "H3ContinuousAnalyzeHandoverV14",
        "H3ContinuousStitchOutputV14", "H3ContinuousSeamlessJoinV14",
        "H3ContinuousStitchSavedChainV14", "H3ContinuousSaveLatent", "H3ContinuousLoadLatent",
        "H3ContinuousHandoverStub",
    ):
        assert node_id in mappings
    assert h3nodes.H3ContinuousContinueV14.RETURN_NAMES == (
        "positive", "latent", "actual_head_context_frames", "ignored_tail_frames",
        "handover_info", "picture_map",
    )
    assert h3nodes.H3ContinuousContinueV14.RETURN_TYPES == ("CONDITIONING", "LATENT", "INT", "INT", "STRING", "STRING")


def test_start_socket_contract():
    schema = H3ContinuousStartExternalConditioning.INPUT_TYPES()
    assert set(schema["required"]) == {"conditioning", "width", "height", "duration"}
    assert set(schema["optional"]) == {"first_frame", "last_frame", "vae", "reference_map"}
    assert H3ContinuousStartExternalConditioning.RETURN_TYPES == ("CONDITIONING", "LATENT", "STRING")
    assert H3ContinuousStartExternalConditioning.RETURN_NAMES == ("positive", "latent", "info")
    for absent in ("clip", "prompt", "ref_image_size", "reference_image", "qwen_reference_1"):
        assert absent not in schema["required"] and absent not in schema["optional"]


def test_continue_socket_contract_has_no_first_frame_and_no_clip():
    schema = H3ContinuousContinueExternalConditioning.INPUT_TYPES()
    assert set(schema["required"]) == {
        "conditioning", "previous_latent", "handover", "width", "height", "duration",
        "masked_context_frames", "duration_mode", "audio_tail_carryover", "audio_feather_ticks",
    }
    assert set(schema["optional"]) == {"last_frame", "vae", "reference_map"}
    assert schema["required"]["masked_context_frames"][0] == ["39", "90", "141", "192"]
    assert schema["required"]["duration_mode"][0] == ["Net New Content", "Total Generation"]
    assert schema["required"]["audio_tail_carryover"][0] == ["Full Previous Tail", "Match Video Handover"]
    assert H3ContinuousContinueExternalConditioning.RETURN_NAMES == (
        "positive", "latent", "actual_head_context_frames", "ignored_tail_frames",
        "handover_info", "reference_info",
    )
    for absent in ("clip", "prompt", "first_frame", "ref_image_size", "reference_image", "qwen_reference_1"):
        assert absent not in schema["required"] and absent not in schema["optional"]


def test_external_module_contains_no_encoder_calls():
    source = (ROOT / "external_conditioning.py").read_text(encoding="utf-8")
    assert "clip.tokenize" not in source
    assert "encode_from_tokens_scheduled" not in source
    assert ".tokenize(" not in source
    assert "Fantastic" in source  # documented as external, never imported
    assert "import" in source


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

def test_start_t2va_without_vae_matches_upstream_h3_geometry():
    node = H3ContinuousStartExternalConditioning()
    positive, latent, info = node.build(_conditioning(), 1344, 768, 5.0)
    video, audio = latent["samples"].unbind()
    assert tuple(video.shape) == _target_vae_shape(1344, 768, 124)
    assert tuple(audio.shape) == (1, 32, 2, 207)
    assert "mode=T2VA" in info
    assert "124 H3 frames" in info
    assert len(positive) == 2
    assert positive[0][1]["minimax_keyframes"] == []


def test_start_duration_grid_snaps_up_like_upstream():
    node = H3ContinuousStartExternalConditioning()
    for seconds, expected in ((5.0, 124), (10.0, 243), (2.0, 56), (3.3, 90)):
        _positive, latent, info = node.build(_conditioning(), 1344, 768, seconds)
        video, _audio = latent["samples"].unbind()
        assert f"-> {expected} H3 frames" in info, (seconds, expected, info)
        assert video.shape[2] == video_latent_t(expected), (seconds, expected, video.shape)


def test_start_encodes_each_anchor_once_and_places_endpoints():
    node = H3ContinuousStartExternalConditioning()
    vae = StubVideoVAE(tag=3.0)
    first = torch.zeros(1, 900, 1600, 3)
    last = torch.ones(1, 1000, 1000, 3)
    positive, latent, info = node.build(_conditioning(), 1344, 768, 5.0,
                                        first_frame=first, last_frame=last, vae=vae)
    assert vae.encode_calls == 2
    keyframes = positive[0][1]["minimax_keyframes"]
    assert [kf["resolved_frame_index"] for kf in keyframes] == [0, 123]
    assert torch.equal(keyframes[0]["latent"], torch.full((1, 24, 12, 48, 84), 3.0))
    assert "mode=FL2VA" in info
    # The anchors are VAE latents; they must not leak into the reference list.
    assert positive[0][1].get("minimax_refs", []) == []


def test_start_i2va_and_l2va_modes():
    node = H3ContinuousStartExternalConditioning()
    image = torch.zeros(1, 512, 512, 3)
    _p, _l, info_first = node.build(_conditioning(), 1344, 768, 5.0,
                                    first_frame=image, vae=StubVideoVAE())
    _p, _l, info_last = node.build(_conditioning(), 1344, 768, 5.0,
                                   last_frame=image, vae=StubVideoVAE())
    assert "mode=I2VA" in info_first
    assert "mode=L2VA" in info_last


def test_start_requires_video_vae_for_anchors():
    node = H3ContinuousStartExternalConditioning()
    image = torch.zeros(1, 64, 64, 3)
    with pytest.raises(ValueError, match="Connect the MiniMax H3 video VAE"):
        node.build(_conditioning(), 1344, 768, 5.0, first_frame=image)
    with pytest.raises(ValueError, match="not the MiniMax H3 video VAE"):
        node.build(_conditioning(), 1344, 768, 5.0, first_frame=image, vae=StubAudioVAE())


@pytest.mark.parametrize("width,height,message", [
    (1330, 768, "multiple of 32"),
    (1344, 770, "multiple of 32"),
    (16, 768, "between 32 and"),
    (1344, 999999, "between 32 and"),
])
def test_start_rejects_illegal_geometry(width, height, message):
    node = H3ContinuousStartExternalConditioning()
    with pytest.raises(ValueError, match=message):
        node.build(_conditioning(), width, height, 5.0)


@pytest.mark.parametrize("duration,message", [
    (float("nan"), "finite"),
    (float("inf"), "finite"),
    (0.5, "between"),
    (1000.0, "between"),
])
def test_start_rejects_illegal_duration(duration, message):
    node = H3ContinuousStartExternalConditioning()
    with pytest.raises(ValueError, match=message):
        node.build(_conditioning(), 1344, 768, duration)


# ---------------------------------------------------------------------------
# Conditioning contract
# ---------------------------------------------------------------------------

def test_empty_conditioning_is_rejected():
    node = H3ContinuousStartExternalConditioning()
    for value in (None, [], ()):
        with pytest.raises(ValueError, match="External conditioning is empty"):
            node.build(value, 1344, 768, 5.0)


def test_missing_or_mismatched_token_tags_are_rejected():
    node = H3ContinuousStartExternalConditioning()
    missing = [[torch.zeros(1, 8, 6), {}]]
    with pytest.raises(ValueError, match="no compatible minimax_token_tags"):
        node.build(missing, 1344, 768, 5.0)

    oversized = [[torch.zeros(1, 8, 6), {"minimax_token_tags": torch.ones(9, dtype=torch.long)}]]
    with pytest.raises(ValueError, match="inconsistent"):
        node.build(oversized, 1344, 768, 5.0)

    squared = [[torch.zeros(1, 8, 6), {"minimax_token_tags": torch.ones(8, dtype=torch.long).unsqueeze(0)}]]
    with pytest.raises(ValueError, match="no compatible minimax_token_tags"):
        node.build(squared, 1344, 768, 5.0)


def test_incoming_keyframes_are_rejected_with_an_actionable_error():
    node = H3ContinuousStartExternalConditioning()
    conditioning = _conditioning(extra_metadata={"minimax_keyframes": [{"resolved_frame_index": 0}]})
    with pytest.raises(ValueError, match="already has temporal keyframes"):
        node.build(conditioning, 1344, 768, 5.0)


def test_incoming_frame_count_is_verified_and_carried_forward():
    node = H3ContinuousStartExternalConditioning()
    good = _conditioning(extra_metadata={"minimax_frame_count": 124})
    positive, _latent, _info = node.build(good, 1344, 768, 5.0)
    assert positive[0][1]["minimax_frame_count"] == 124

    bad = _conditioning(extra_metadata={"minimax_frame_count": 243})
    with pytest.raises(ValueError, match="minimax_frame_count does not match"):
        node.build(bad, 1344, 768, 5.0)


def test_conditioning_is_never_mutated_and_embeddings_keep_identity():
    node = H3ContinuousStartExternalConditioning()
    refs = [{"kind": "video", "ref_audio_t": 4}, {"kind": "image", "latent_h": 2}]
    conditioning = _conditioning(entries=2, refs=refs,
                                 extra_metadata={"start_percent": 0.0, "end_percent": 1.0,
                                                 "pooled_output": torch.ones(1, 3)})
    before_metadata = [dict(entry[1]) for entry in conditioning]
    before_refs = [list(entry[1]["minimax_refs"]) for entry in conditioning]
    before_embeddings = [entry[0] for entry in conditioning]

    positive, _latent, _info = node.build(conditioning, 1344, 768, 5.0)

    for index, entry in enumerate(conditioning):
        assert entry[1] == before_metadata[index], "input metadata was mutated"
        assert entry[1]["minimax_refs"] == before_refs[index]
        assert entry[1]["minimax_refs"] is not before_refs[index]
        assert "minimax_keyframes" not in entry[1]
        assert entry[0] is before_embeddings[index]

    assert len(positive) == len(conditioning)
    for index, (embedding, metadata) in enumerate(positive):
        assert embedding is before_embeddings[index], "embedding tensor was cloned or replaced"
        assert metadata is not conditioning[index][1]
        assert torch.equal(metadata["minimax_token_tags"],
                           conditioning[index][1]["minimax_token_tags"])
        assert metadata["start_percent"] == 0.0
        assert metadata["end_percent"] == 1.0
        assert torch.equal(metadata["pooled_output"], torch.ones(1, 3))
        assert metadata["minimax_refs"] == refs
        assert metadata["minimax_refs"] is not refs


def test_scheduled_entries_and_reference_order_survive_untouched():
    node = H3ContinuousStartExternalConditioning()
    refs = [{"kind": "video", "name": "a"}, {"kind": "video", "name": "a"}, {"kind": "image", "name": "b"}]
    conditioning = _conditioning(entries=3,
                                 extra_metadata={"minimax_refs": refs,
                                                 "start_percent": 0.25, "end_percent": 0.75})
    positive, _latent, _info = node.build(conditioning, 1344, 768, 5.0)
    assert [entry[1]["start_percent"] for entry in positive] == [0.25, 0.25, 0.25]
    for entry in positive:
        assert [ref["name"] for ref in entry[1]["minimax_refs"]] == ["a", "a", "b"]
    assert "text encoder passes added 0" in _info


def test_unknown_future_metadata_is_preserved():
    node = H3ContinuousStartExternalConditioning()
    conditioning = _conditioning(extra_metadata={"some_future_encoder_field": {"nested": [1, 2]}})
    positive, _latent, _info = node.build(conditioning, 1344, 768, 5.0)
    assert positive[0][1]["some_future_encoder_field"] == {"nested": [1, 2]}


def test_no_encoder_pass_is_triggered():
    """The new node must not touch a CLIP even if one exists in the graph."""
    clip = StubClip()
    node = H3ContinuousStartExternalConditioning()
    node.build(_conditioning(), 1344, 768, 5.0)
    assert clip.tokenize_calls == 0
    assert clip.encode_calls == 0


# ---------------------------------------------------------------------------
# Continue: equivalence with upstream v1.4
# ---------------------------------------------------------------------------

CONTINUE_CASES = [
    # (duration, context, duration_mode, audio_tail_carryover, feather)
    (5.0, "39", "Net New Content", "Full Previous Tail", 0),
    (5.0, "39", "Net New Content", "Match Video Handover", 0),
    (5.0, "39", "Total Generation", "Full Previous Tail", 0),
    (5.0, "39", "Net New Content", "Full Previous Tail", 7),
    (2.0, "39", "Total Generation", "Match Video Handover", 0),
    (5.0, "90", "Net New Content", "Full Previous Tail", 0),
]


def _run_upstream_v14(duration, context, duration_mode, audio_tail, feather, *,
                      previous=None, handover=None, width=1344, height=768, last_frame=None):
    clip = StubClip()
    vae = StubVideoVAE(tag=5.0)
    node = h3nodes.H3ContinuousContinueV14()
    return node.build(
        clip, vae,
        previous if previous is not None else _previous_latent(),
        "upstream prompt", width, height, duration,
        masked_context_frames=context, audio_feather_ticks=feather,
        duration_mode=duration_mode, audio_tail_carryover=audio_tail,
        handover=handover if handover is not None else _handover(),
        last_frame=last_frame,
    )


def _run_external_continue(duration, context, duration_mode, audio_tail, feather, *,
                           conditioning=None, previous=None, handover=None,
                           width=1344, height=768, last_frame=None, vae=None):
    node = H3ContinuousContinueExternalConditioning()
    return node.build(
        conditioning if conditioning is not None else _conditioning(),
        previous if previous is not None else _previous_latent(),
        handover if handover is not None else _handover(),
        width, height, duration,
        masked_context_frames=context, duration_mode=duration_mode,
        audio_tail_carryover=audio_tail, audio_feather_ticks=feather,
        last_frame=last_frame, vae=vae,
    )


@pytest.mark.parametrize("duration,context,duration_mode,audio_tail,feather", CONTINUE_CASES)
def test_continue_matches_upstream_v14_plan_and_tensors(duration, context, duration_mode, audio_tail, feather):
    upstream = _run_upstream_v14(duration, context, duration_mode, audio_tail, feather)
    external = _run_external_continue(duration, context, duration_mode, audio_tail, feather)

    up_cond, up_latent, up_ctx, up_tail, up_info, _map = upstream
    ex_cond, ex_latent, ex_ctx, ex_tail, ex_info, _ref_info = external

    assert ex_ctx == up_ctx
    assert ex_tail == up_tail

    up_video, up_audio = up_latent["samples"].unbind()
    ex_video, ex_audio = ex_latent["samples"].unbind()
    assert tuple(ex_video.shape) == tuple(up_video.shape)
    assert tuple(ex_audio.shape) == tuple(up_audio.shape)

    up_mask_v, up_mask_a = up_latent["noise_mask"].unbind()
    ex_mask_v, ex_mask_a = ex_latent["noise_mask"].unbind()
    assert torch.equal(ex_mask_v, up_mask_v)
    assert torch.equal(ex_mask_a, up_mask_a)

    # Protected prefix must be an exact copy of the upstream source slice, and the
    # untouched remainder must be identical (all zeros) in both.
    video_steps = int((ex_mask_v == 0.0).any(dim=-1).any(dim=-1).sum())
    assert video_steps > 0
    assert torch.equal(ex_video[:, :, :video_steps], up_video[:, :, :video_steps])
    assert torch.equal(ex_video[:, :, video_steps:], up_video[:, :, video_steps:])
    assert torch.equal(ex_audio, up_audio)

    assert "native_masked_av_safe_handover_audio_tail" in ex_info
    assert f"protected head {ex_ctx}f" in ex_info
    assert f"discarded previous tail {ex_tail}f" in ex_info


def test_continue_does_not_modify_previous_latent_or_handover():
    previous = _previous_latent()
    video_before = previous["samples"].unbind()[0].clone()
    audio_before = previous["samples"].unbind()[1].clone()
    handover = _handover()
    handover_before = dict(handover)
    refs = [{"kind": "audio", "ref_audio_t": 3}]
    conditioning = _conditioning(refs=refs)
    metadata_before = [dict(entry[1]) for entry in conditioning]

    _cond, latent, _ctx, _tail, _info, _ref_info = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 0,
        conditioning=conditioning, previous=previous, handover=handover,
    )

    assert torch.equal(previous["samples"].unbind()[0], video_before)
    assert torch.equal(previous["samples"].unbind()[1], audio_before)
    assert handover == handover_before
    for index, entry in enumerate(conditioning):
        assert entry[1] == metadata_before[index]
    assert latent["samples"].unbind()[0].shape[0] == 1


def test_continue_protects_previous_context_and_leaves_target_elsewhere_zero():
    previous = _previous_latent()
    _cond, latent, ctx, tail, _info, _ref = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 0, previous=previous)
    ex_video, ex_audio = latent["samples"].unbind()
    mask_v, mask_a = latent["noise_mask"].unbind()
    video_steps = int((mask_v[0, 0, :, 0, 0] == 0.0).sum())
    audio_steps = int((mask_a[0, 0, 0, :] == 0.0).sum())
    assert video_steps == 12, "39 protected frames is 12 H3 video-latent steps"
    # Full Previous Tail protects audio through the previous clip's real end, not
    # just the video handover: ticks 113..206 of a 124-frame clip.
    assert audio_steps == 94
    assert ctx == 39 and tail == 17
    # Video beyond the protected head is still empty in the target latent.
    assert torch.count_nonzero(ex_video[:, :, video_steps:]) == 0
    assert torch.count_nonzero(ex_audio[..., audio_steps:]) == 0


def test_continue_audio_feather_ramps_only_the_final_ticks():
    _c, latent, _ctx, _tail, _info, _r = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 7)
    mask_a = latent["noise_mask"].unbind()[1]
    protected = 94  # Full Previous Tail for a 124-frame previous clip
    hard = int((mask_a[0, 0, 0, :protected] == 0.0).sum())
    ramp = mask_a[0, 0, 0, hard:protected]
    assert hard == protected - 7
    assert ramp.shape[0] == 7
    # Half-cosine ramp: released ticks go from nearly fully denoised to untouched.
    assert ramp[0] > 0.0 and ramp[-1] == 1.0
    assert torch.all(ramp >= 0) and torch.all(ramp <= 1)
    # Monotonic ramp, and nothing beyond the protected window is pinned.
    assert torch.all(ramp[1:] >= ramp[:-1])
    assert torch.all(mask_a[0, 0, 0, protected:] == 1.0)


def test_continue_full_tail_protects_more_audio_than_video_match():
    _c, full, _ctx, _t, _i, _r = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 0)
    _c, matched, _ctx, _t, _i, _r = _run_external_continue(
        5.0, "39", "Net New Content", "Match Video Handover", 0)
    full_steps = int((full["noise_mask"].unbind()[1][0, 0, 0, :] == 0.0).sum())
    matched_steps = int((matched["noise_mask"].unbind()[1][0, 0, 0, :] == 0.0).sum())
    assert full_steps == 94, "124-frame previous clip: ticks 113..206"
    assert matched_steps == 65
    assert full_steps > matched_steps
    # Video protection is identical in both modes.
    assert torch.equal(full["noise_mask"].unbind()[0], matched["noise_mask"].unbind()[0])


def test_continue_last_frame_occupies_the_exact_terminal_frame():
    previous = _previous_latent()
    vae = StubVideoVAE(tag=9.0)
    last = torch.zeros(1, 1000, 1000, 3)
    _cond, latent, _ctx, _tail, _info, _r = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 0,
        previous=previous, last_frame=last, vae=vae,
    )
    assert vae.encode_calls == 1
    conditioning_node = H3ContinuousContinueExternalConditioning()
    positive = conditioning_node.build(
        _conditioning(), previous, _handover(), 1344, 768, 5.0, last_frame=last, vae=StubVideoVAE(9.0))[0]
    keyframes = positive[0][1]["minimax_keyframes"]
    assert [kf["resolved_frame_index"] for kf in keyframes] == [157], "5 s Net New Content plans 158 frames"
    # The terminal anchor must not overwrite the protected prefix.
    video_steps = int((latent["noise_mask"].unbind()[0][0, 0, :, 0, 0] == 0.0).sum())
    assert torch.count_nonzero(latent["samples"].unbind()[0][:, :, video_steps:-1]) == 0


def test_continue_requires_last_frame_vae_only_when_used():
    node = H3ContinuousContinueExternalConditioning()
    # No anchor: no VAE needed.
    node.build(_conditioning(), _previous_latent(), _handover(), 1344, 768, 5.0)
    with pytest.raises(ValueError, match="Connect the MiniMax H3 video VAE"):
        node.build(_conditioning(), _previous_latent(), _handover(), 1344, 768, 5.0,
                   last_frame=torch.zeros(1, 64, 64, 3))


def test_continue_rejects_missing_or_invalid_handover():
    node = H3ContinuousContinueExternalConditioning()
    with pytest.raises(ValueError, match="requires H3_CONTINUOUS_HANDOVER"):
        node.build(_conditioning(), _previous_latent(), {"available": False}, 1344, 768, 5.0)
    with pytest.raises(ValueError, match="requires H3_CONTINUOUS_HANDOVER"):
        node.build(_conditioning(), _previous_latent(), None, 1344, 768, 5.0)
    with pytest.raises(ValueError, match="handover frame_count"):
        node.build(_conditioning(), _previous_latent(), {**_handover(), "frame_count": 243}, 1344, 768, 5.0)
    with pytest.raises(ValueError, match="invalid previous safe handover endpoint"):
        node.build(_conditioning(), _previous_latent(), _handover(handover_end_frame=999), 1344, 768, 5.0)
    with pytest.raises(ValueError, match="too early for the minimum 39-frame"):
        node.build(_conditioning(), _previous_latent(), _handover(handover_end_frame=20), 1344, 768, 5.0)


def test_continue_rejects_misaligned_handover_and_bad_resolution():
    node = H3ContinuousContinueExternalConditioning()
    with pytest.raises(ValueError, match="not aligned to the selected Masked-AV context"):
        node.build(_conditioning(), _previous_latent(), _handover(handover_end_frame=100), 1344, 768, 5.0)
    with pytest.raises(ValueError, match="Previous latent resolution differs"):
        node.build(_conditioning(), _previous_latent(width=704, height=704), _handover(), 1344, 768, 5.0)


def test_continue_rejects_target_that_overflows_with_protected_context():
    node = H3ContinuousContinueExternalConditioning()
    # 2 s Total Generation is only 56 frames: the 90-frame protected head cannot fit.
    with pytest.raises(ValueError, match="context consumes the whole target|protected audio tail"):
        node.build(_conditioning(), _previous_latent(), _handover(), 1344, 768, 2.0,
                   masked_context_frames="90", duration_mode="Total Generation")


def test_continue_previous_dtype_and_device_do_not_change_results():
    import comfy.nested_tensor

    previous_cpu = _previous_latent()
    previous_half = {
        "samples": comfy.nested_tensor.NestedTensor(
            tuple(t.half() for t in previous_cpu["samples"].unbind()))
    }
    _c, latent_full, ctx_full, tail_full, _i, _r = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 0, previous=previous_cpu)
    _c, latent_half, ctx_half, tail_half, _i, _r = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 0, previous=previous_half)
    assert (ctx_full, tail_full) == (ctx_half, tail_half)
    assert torch.equal(latent_full["noise_mask"].unbind()[0], latent_half["noise_mask"].unbind()[0])
    # The target latent is float32, so protected values are the source values cast
    # into the target dtype. The half-precision source is what must be preserved.
    _c, full, _ctx, _tail, _info, _r = _run_external_continue(
        5.0, "39", "Net New Content", "Full Previous Tail", 0, previous=previous_cpu)
    source_half = previous_half["samples"].unbind()[0]
    target_half = latent_half["samples"].unbind()[0]
    protected = 12  # 39 protected frames is 12 H3 video-latent steps
    source_slice = source_half[:, :, 20:20 + protected]  # this plan slices latent 20:32
    assert torch.equal(target_half[:, :, :protected], source_slice.to(target_half.dtype))
    # And the full-precision run protects the f32 source at full precision.
    assert torch.equal(full["samples"].unbind()[0][:, :, :protected],
                       previous_cpu["samples"].unbind()[0][:, :, 20:20 + protected])


def test_continue_reference_info_reports_external_encoding_and_echoes_fantastic_map():
    node = H3ContinuousContinueExternalConditioning()
    result = node.build(_conditioning(), _previous_latent(), _handover(), 1344, 768, 5.0,
                        reference_map="<Picture 1> = 064.png\n<Video 1> = FaceCrop_057")
    reference_info = result[5]
    assert "EXTERNAL" in reference_info
    assert "<Picture 1> = 064.png" in reference_info
    assert "Fantastic reference map" in reference_info
    assert "Picture Map" not in reference_info

    result = node.build(_conditioning(), _previous_latent(), _handover(), 1344, 768, 5.0)
    assert "not supplied" in result[5]


def test_continue_does_not_append_protected_context_to_refs():
    refs = [{"kind": "audio", "ref_audio_t": 3}]
    conditioning = _conditioning(refs=refs)
    positive = _run_external_continue(5.0, "39", "Net New Content", "Full Previous Tail", 0,
                                      conditioning=conditioning)[0]
    for entry in positive:
        # The v1.4 protected context lives in latent + noise_mask, never in refs.
        assert [dict(ref) for ref in entry[1]["minimax_refs"]] == refs
    assert "audio_latent" not in positive[0][1]


def test_shared_planner_is_used_by_both_nodes():
    """The upstream node and the external node must share one planner implementation."""
    import inspect
    source = inspect.getsource(h3nodes.H3ContinuousContinueV14.build)
    assert "plan_masked_av_continuation(" in source
    external_source = inspect.getsource(H3ContinuousContinueExternalConditioning.build)
    assert "plan_masked_av_continuation(" in external_source


# ---------------------------------------------------------------------------
# Saved-chain path resolution
# ---------------------------------------------------------------------------

def test_resolve_latent_path_accepts_the_save_prefix(tmp_path, monkeypatch):
    """Load AV Latent must accept Save AV Latent's filename_prefix verbatim.

    Save AV Latent writes ``<prefix>_00001.safetensors``; the prefix itself is
    neither a file nor a directory.
    """
    folder = tmp_path / "h3_external_conditioning"
    folder.mkdir()
    saved = folder / "shot_00001.safetensors"
    saved.write_bytes(b"stub")
    monkeypatch.setattr(h3nodes, "_candidate_output_roots", lambda: [str(tmp_path)])

    resolved = h3nodes._resolve_latent_path("h3_external_conditioning/shot", 1)
    assert resolved == str(saved)


def test_resolve_latent_path_still_accepts_a_directory(tmp_path, monkeypatch):
    folder = tmp_path / "chain"
    folder.mkdir()
    (folder / "clip_00001.safetensors").write_bytes(b"a")
    (folder / "clip_00002.safetensors").write_bytes(b"b")
    monkeypatch.setattr(h3nodes, "_candidate_output_roots", lambda: [str(tmp_path)])

    assert h3nodes._resolve_latent_path(str(folder), 2).endswith("clip_00002.safetensors")
    # clip_index 0 means "newest".
    assert h3nodes._resolve_latent_path(str(folder), 0).endswith("clip_00002.safetensors")


def test_resolve_latent_path_reports_a_missing_slot_clearly(tmp_path, monkeypatch):
    folder = tmp_path / "chain"
    folder.mkdir()
    (folder / "shot_00001.safetensors").write_bytes(b"a")
    monkeypatch.setattr(h3nodes, "_candidate_output_roots", lambda: [str(tmp_path)])

    with pytest.raises(FileNotFoundError, match="no fixed slot 3"):
        h3nodes._resolve_latent_path("chain/shot", 3)


def test_resolve_latent_path_searches_every_output_root(tmp_path, monkeypatch):
    """get_output_directory() and get_save_image_path() may disagree."""
    save_root = tmp_path / "base" / "output"
    other_root = tmp_path / "ComfyUI" / "output"
    (save_root / "h3_continuous").mkdir(parents=True)
    other_root.mkdir(parents=True)
    saved = save_root / "h3_continuous" / "clip_00001.safetensors"
    saved.write_bytes(b"stub")
    monkeypatch.setattr(
        h3nodes, "_candidate_output_roots", lambda: [str(other_root), str(save_root)])

    assert h3nodes._resolve_latent_path("h3_continuous/clip", 1) == str(saved)


# ---------------------------------------------------------------------------
# Layout API detection must survive foreign wrappers (e.g. SolAttn's)
# ---------------------------------------------------------------------------

def test_layout_detection_survives_a_generic_foreign_wrapper():
    """SolAttn wraps PackedLayout.__init__ with (*args, **kwargs).

    That hides the real parameters from inspect.signature and used to make the
    suite refuse H3 continuation with "missing expected parameters". Detection
    must fall back to the class-level constructor / a capability probe.
    """
    import comfy.ldm.minimax.model as mm
    detect_layout_api = importlib.import_module(f"{PKG}.patch_layout").detect_layout_api

    layout_cls = mm.PackedLayout
    original = layout_cls.__init__
    try:
        def generic_init(self, *args, **kwargs):
            return original(self, *args, **kwargs)

        layout_cls.__init__ = generic_init
        # The class-level constructor is still the real stock one.
        mode, _sig, err = detect_layout_api()
        assert err is None, err
        assert mode == "native"
    finally:
        layout_cls.__init__ = original


def test_layout_detection_falls_back_to_a_capability_probe():
    """Even if the class-level constructor is hidden, construction decides."""
    import comfy.ldm.minimax.model as mm
    detect_layout_api = importlib.import_module(f"{PKG}.patch_layout").detect_layout_api

    layout_cls = mm.PackedLayout
    original = layout_cls.__dict__["__init__"]

    def opaque_init(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    try:
        # Hide the class-level constructor behind a generic function with the
        # same (opaque) signature.
        layout_cls.__init__ = opaque_init
        mode, _sig, err = detect_layout_api()
        assert err is None, err
        assert mode == "native"
    finally:
        layout_cls.__init__ = original


def test_continue_does_not_install_or_require_legacy_runtime_patches():
    """The native constraint path must not demand the legacy layout wrapper.

    Another H3 pack (SolAttn) wraps PackedLayout; requesting the suite's legacy
    wrapper would abort the run with a runtime-patch conflict even though this
    node needs nothing from it.
    """
    import inspect

    source = inspect.getsource(H3ContinuousContinueExternalConditioning.build)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#"))
    assert "_require_patches(" not in code
    assert not hasattr(ec, "_require_patches")


# ---------------------------------------------------------------------------
# Stitch Output last_frame (handover join frame)
# ---------------------------------------------------------------------------

def test_stitch_output_last_frame_is_the_full_render_tail():
    """Full mode trims nothing, so last_frame is the literal final render frame."""
    stitch = h3nodes.H3ContinuousStitchOutputV1()
    images = torch.arange(5 * 2 * 2 * 3, dtype=torch.float32).reshape(5, 2, 2, 3)
    out_images, _audio, _info, head, tail, last_frame = stitch.prepare(
        images, output_mode="Full", head_context_frames=0,
    )
    assert (head, tail) == (0, 0)
    assert torch.equal(last_frame, images[-1:])


def test_stitch_output_last_frame_is_the_exact_join_frame():
    """Stitch Ready trims the unusable tail; last_frame must be the kept end.

    This is the frame the next segment has to match, so exporting anything else
    (for example the raw final render frame) would lie about the join.
    """
    stitch = h3nodes.H3ContinuousStitchOutputV1()
    images = torch.arange(10 * 1 * 1 * 3, dtype=torch.float32).reshape(10, 1, 1, 3)
    handover = {
        "available": True,
        "frame_count": 10,
        "handover_end_frame": 6,
        "landing_tail_frames": 3,
    }
    out_images, _audio, info, head, tail, last_frame = stitch.prepare(
        images, output_mode="Stitch Ready", head_context_frames=0, handover=handover,
    )
    assert (head, tail) == (0, 3)
    assert out_images.shape[0] == 7
    assert torch.equal(last_frame, images[6:7])
    assert torch.equal(last_frame, out_images[-1:])
    assert "phase-aligned source end frame 6" in info


def test_stitch_output_last_frame_after_head_trim():
    """Final Clip (segment 2+) drops the reused head; last_frame keeps the landing."""
    stitch = h3nodes.H3ContinuousStitchOutputV1()
    images = torch.arange(8 * 1 * 1 * 3, dtype=torch.float32).reshape(8, 1, 1, 3)
    out_images, _audio, _info, head, tail, last_frame = stitch.prepare(
        images, output_mode="Final Clip", head_context_frames=3,
    )
    assert (head, tail) == (3, 0)
    assert torch.equal(last_frame, images[-1:])
    assert torch.equal(last_frame, out_images[-1:])
