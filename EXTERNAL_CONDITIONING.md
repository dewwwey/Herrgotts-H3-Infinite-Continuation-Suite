# H3 External Conditioning (Fantastic RefMods)

Two nodes that build MiniMax H3 generation latents and temporal constraints from
**already encoded** H3 `CONDITIONING`, so the continuation suite no longer needs
to own a text encoder.

| | |
| --- | --- |
| Category | `Herrgotts H3 Infinite Continuation Suite/External Conditioning` |
| Nodes | `H3ContinuousStartExternalConditioning`, `H3ContinuousContinueExternalConditioning` |
| Display names | **H3 Start — External Conditioning**, **H3 Continue — External Conditioning** |
| Added in | fork feature branch `feat/external-conditioning` (base upstream `4b1edd6`) |
| Shared planner | `plan_masked_av_continuation()` in `nodes.py`, used by both these nodes and upstream `H3ContinuousContinueV14` |

## Why

The upstream v1.4 `Start`/`Continue` nodes take a `clip`, a `prompt` and a
`reference_image`/`qwen_reference_*` and run their own tokenization. With
Fantastic's RefMod Text Encode already owning prompt + RefMod + stored reference
images, that second encoder pass is duplicate work that can also re-present (and
therefore duplicate) the same reference bundle. These nodes keep the suite's
v1.4 continuation semantics but consume conditioning that is already built.

## Division of responsibility

| Owner | Responsibility |
| --- | --- |
| **Fantastic — RefMod Stack / Media Loader / Prompt Builder** | RefMod bundle, stored reference images, media references, prompt text, `reference_map` |
| **Fantastic — RefMod Text Encode** | The single tokenization + `clip.encode_from_tokens_scheduled` pass; emits `CONDITIONING` carrying `minimax_token_tags` and `minimax_refs` |
| **H3 Start — External Conditioning** | Segment 1 generation latent + optional First/Last temporal anchors |
| **H3 Continue — External Conditioning** | Segment N latent + the v1.4 native Masked-AV protected context, per-stream denoise masks and safe handover slice |
| **H3 Save/Load AV Latent, Auto Handover, Stitch** | Disk state, freeze analysis, output trimming, final stitching |

Facts that follow from that split:

* **No tokenizer is called.** Neither node imports Fantastic and neither node
  contains a `tokenize` / `encode_from_tokens_scheduled` call.
* **Fantastic's `latent` output is unused.** Its `length` does not decide the
  segment length; `duration` on these nodes does. Set Fantastic's `width`,
  `height` and `length` to your actual generation geometry and to at least the
  segment length, because its own media-reference trimming still uses `length`.
* **Fantastic video-edit mode is out of scope.** That mode returns an edit
  latent carrying mask/protection state; discarding it would discard the
  intended edit. Do not use these nodes for video editing.
* **Neither node keeps state.** No disk I/O, no module-level cache, no tensors
  retained between executions.

## Endpoint images are temporal anchors, not Qwen pictures

`first_frame` / `last_frame` accept an `IMAGE` and are VAE-encoded straight into
the target latent as an H3 keyframe. They are **not** shown to the text encoder
and therefore do **not** become `<Picture N>`.

Consequences:

* If the prompt needs the encoder to *see* an endpoint image, present it through
  Fantastic (Media Loader / Media slot) before encoding and refer to the label
  Fantastic assigns in its `reference_map`.
* Fantastic's ordinary media inputs can additionally create **persistent DiT
  reference blocks** that ride through every sampling step. That is a different
  mechanism from a temporal-only anchor; do not expect the two to be
  interchangeable.
* Continue has no `first_frame` on purpose: the protected previous context is
  this segment's starting state.

## Socket reference

### H3 Start — External Conditioning (`build`)

| Input | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `conditioning` | CONDITIONING | yes | — | Fantastic `conditioning`. Must carry H3 `minimax_token_tags`, and must not already have temporal keyframes. |
| `width` | INT | yes | 1344 | Multiple of 32; must match the geometry Fantastic encoded its references at. |
| `height` | INT | yes | 768 | Multiple of 32. |
| `duration` | FLOAT | yes | 5.0 | Total first-segment duration, 2.0–150.0 s. Snapped **up** to H3's 17k+5 frame grid at 24 fps (5.0 s → 124 frames). |
| `first_frame` | IMAGE | no | — | Optional temporal start anchor (frame 0). Requires `vae`. |
| `last_frame` | IMAGE | no | — | Optional temporal end anchor (frame `frame_count-1`). Requires `vae`. |
| `vae` | VAE | no | — | MiniMax H3 video VAE. Required only when an anchor is connected. |
| `reference_map` | STRING | no | — | Diagnostic passthrough from Fantastic; echoed in `info`, never parsed. |

Outputs: `positive` (CONDITIONING), `latent` (LATENT), `info` (STRING).

Connecting neither, either or both anchors gives T2VA, I2VA / L2VA and FL2VA.

### H3 Continue — External Conditioning (`build`)

| Input | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `conditioning` | CONDITIONING | yes | — | As above. |
| `previous_latent` | LATENT | yes | — | Previous segment's **full sampled** AV latent (Load AV Latent), not trimmed render frames. |
| `handover` | H3_CONTINUOUS_HANDOVER | yes | — | Previous segment's v1.4 safe-handover metadata. |
| `width` / `height` | INT | yes | 1344 / 768 | Must match the previous video latent grid (checked). |
| `duration` | FLOAT | yes | 5.0 | Interpreted by `duration_mode`; 2.0–150.0 s. |
| `masked_context_frames` | ENUM | yes | `"39"` | `"39"`, `"90"`, `"141"`, `"192"`. Exact joint H3 video boundaries. |
| `duration_mode` | ENUM | yes | `Net New Content` | Or `Total Generation` (legacy whole-clip meaning). |
| `audio_tail_carryover` | ENUM | yes | `Full Previous Tail` | Or `Match Video Handover`. |
| `audio_feather_ticks` | INT | yes | 0 | Advanced; keep 0 for hard dialogue protection. |
| `last_frame` | IMAGE | no | — | Optional future endpoint anchor. Requires `vae`. |
| `vae` | VAE | no | — | Required only when `last_frame` is supplied. |
| `reference_map` | STRING | no | — | As above; echoed in `reference_info`. |

Outputs: `positive` (CONDITIONING), `latent` (LATENT),
`actual_head_context_frames` (INT), `ignored_tail_frames` (INT),
`handover_info` (STRING), `reference_info` (STRING).

The first five outputs keep the upstream `H3ContinuousContinueV14` meanings.
`reference_info` states explicitly that text/reference encoding is external and
echoes the supplied Fantastic `reference_map`; it never invents a Herrgott
Picture Map.

## Conditioning contract

Both nodes follow the same rules:

1. **Every entry** in the conditioning list is processed, including scheduled
   entries — not just entry 0.
2. Each entry gets a **new metadata dict**; the caller's list, metadata and
   reference containers are never mutated.
3. The **embedding tensor is passed through by identity** — never cloned,
   re-cast or moved.
4. `minimax_token_tags` are preserved exactly with their embedding.
5. Unrelated metadata (`start_percent`, `end_percent`, pooled outputs, unknown
   future encoder fields) is preserved.
6. Each entry's own `minimax_refs` keeps its order and contents (containers are
   copied before being replaced).
7. New temporal anchors go into `minimax_keyframes` with the upstream index
   structure: First is frame 0, Last is `frame_count - 1`.
8. The v1.4 protected AV context lives in `latent["samples"]` and
   `latent["noise_mask"]`. It is **never** appended to `minimax_refs`.

**Conflict policy.** These nodes expect raw reference conditioning. If the
incoming conditioning already carries temporal keyframes, they raise
`Incoming conditioning already has temporal keyframes; supply raw reference
conditioning and connect anchors here.` A pre-existing `minimax_frame_count` is
verified against the planned target and rejected on mismatch; harmless empty
containers are preserved.

## Actionable errors

| Situation | Message |
| --- | --- |
| Empty / non-sequence conditioning | `External conditioning is empty; connect an H3 text-encode conditioning output.` |
| Entry without usable tags | `Conditioning entry N has no compatible minimax_token_tags; use an H3 encoder (Fantastic RefMod Text Encode or MiniMax H3 Reference to Video).` |
| Tags length ≠ embedding sequence | `Conditioning entry N has X minimax_token_tags for an embedding sequence of Y positions; …` |
| Incoming keyframes | `Incoming conditioning already has temporal keyframes; supply raw reference conditioning and connect anchors here.` |
| Incoming frame count mismatch | `Incoming minimax_frame_count does not match the planned target frame count. …` |
| Anchor without a VAE | `Connect the MiniMax H3 video VAE when First/Last Frame anchors are supplied.` |
| Wrong VAE kind | `The VAE connected for the First/Last Frame anchor is not the MiniMax H3 video VAE. …` |
| Geometry / duration | `width must be a multiple of 32 …`, `duration must be between 2.0 and 150.0 seconds …` |
| Continue resolution mismatch | `Previous latent resolution differs from the requested target resolution. …` |
| Bad / missing handover | Upstream v1.4 messages, e.g. `v1.4 Continue requires H3_CONTINUOUS_HANDOVER metadata for the previous clip. …` |
| Context overflows target | Upstream messages, e.g. `h3_continuous: protected audio tail would consume the whole target audio latent. …` |

## The split three-job segment workflow

`examples/` ships the split as three separately submittable graphs:

| File | Job | What it does |
| --- | --- | --- |
| `Herrgotts_H3_Infinite_v1.5_ExternalConditioning_01_Start.json` | Job 1 | Fantastic → **Start** → sampler → decode → Auto Handover → Save AV Latent (clip 1) → Stitch Ready render → **save `last_frame`** |
| `Herrgotts_H3_Infinite_v1.5_ExternalConditioning_02_Continue.json` | Job 2 | Load AV Latent (clip 1) → Fantastic (with clip 1's last frame as `<Picture 1>`) → **Continue** → sampler → decode → Auto Handover → Save AV Latent (clip 2) → Final Clip render |
| `Herrgotts_H3_Infinite_v1.5_ExternalConditioning_03_Stitch.json` | Job 3 | **Stitch Saved Chain v1.4** joins the saved clip files into the final video |

The same graphs are also available as API-format prompts for headless running:

```
workflows/job1_start_api.json
workflows/job2_continue_api.json
workflows/job3_stitch_api.json
```

Run them in order (jobs 1 and 2 write to the same `latent_prefix`,
`h3_external_conditioning/shot`). Job 2 needs clip 1's saved latent on disk and
job 3 needs both, so each job must complete before the next starts. The stitch
is its own job because `H3ContinuousStitchSavedChainV14` has no data dependency
on the Save AV Latent node: inside one prompt it would execute before clip 2
exists and silently skip. Its `IS_CHANGED` is a saved-chain file fingerprint so
re-queued stitches re-run once the chain is complete instead of returning a
cached "incomplete" result.

Handover between the jobs is **entirely on disk**:

```
job 1 sampler ─▶ Save AV Latent clip 1 ─┐
job 1 Auto Handover ───────────────────┘  h3_external_conditioning/shot_00001.safetensors
                                          (safetensors metadata: handover_json + head_context_frames)
job 1 Stitch Ready ─▶ last_frame output ─▶ SaveImage
                                          h3_external_conditioning/handover_last_frame_00001_.png
job 2 Media Loader ─▶ <Picture 1> = last frame ─▶ Fantastic ─▶ Continue ─▶ segment 2
job 2 ─▶ shot_00002.safetensors
job 3 Stitch Saved Chain v1.4 ─▶ video/H3_External_Conditioning_final_*.mp4
```

`SaveImage` appends a frame counter on every run; delete stale
`output/h3_external_conditioning/handover_last_frame_*.png` before re-running
job 1 so job 2's Media Loader path (`handover_last_frame_00001_..png [output]`)
keeps pointing at the fresh frame. Fantastic's media resolver accepts
`[output]`-annotated paths (input, output and temp directories are all
allowed roots).

No loop-carried in-memory state is involved: each job is a fresh ComfyUI
prompt, so a long chain does not grow the executor cache.

## Scene-continuity handover: the last frame as a picture

The masked-AV context anchors the **head** of the continuation (its first
39 protected frames decode as the previous clip), but the *body* is generated
from conditioning alone. If the text encoder never sees the ongoing scene, a
continuation segment can drift or hard-cut into an unrelated scene mid-body —
observed live as a beach scene that held for ~0.7 s after the seam and then cut
to an indoor close-up that matched neither the prompt nor the reference
character.

The fix is the same mechanism the stock suite uses for its own Continue node:
give the next segment's **text encoder** the previous clip's actual join frame
as a picture. The suite therefore exposes one new output:

* **`H3ContinuousStitchOutputV1/V11/V14` → `last_frame` (IMAGE)** — the exact
  post-trim join frame (the last frame this clip contributes after head/tail
  trimming). This is the frame the successor must match; the raw final render
  frame is *not* the same when a freeze/no-lock tail was trimmed.

Job 1 saves it (`SaveImage`), job 2's Media Loader loads it back as a picture,
and the segment prompt addresses it explicitly (for example `<Picture 1> is the
exact last frame of [Shot 1] ... [Shot 2] continues from it with no cut`).
This stays within the external-conditioning contract: the encode still happens
in Fantastic, the new nodes still perform zero tokenize/encode calls.

The split workflows' segment prompts also encode camera-move continuity: the
truck in starts in [Shot 1] and [Shot 2] continues the same move ("no cut")
until it stops on the close-up, instead of cutting to a new framing.

Two related findings from the live debugging session:

* The external **Continue** node deliberately does not call
  `_require_patches()`. The native path needs only core masked-AV support plus
  the stock keyframe layout, and requesting the suite's legacy layout wrapper
  would collide with other packs that wrap `PackedLayout` (observed with
  SolAttn's generic signature-hiding wrapper, which
  `patch_layout.detect_layout_api()` now also survives).
* The v1.4 native Masked-AV protection itself worked: with the picture handover
  in place the stitched video contains no hard cut anywhere (ffmpeg scene-score
  scan), and the exported `last_frame` differs from the stitched seam frame by
  ~1% mean channel value (codec/crossfade noise).

## Prerequisites

* A current ComfyUI with native MiniMax H3 AV-mask support (PR #15375). The
  Continue node runs the suite's live capability check and refuses to run
  without it.
* Fantastic installed and enabled (`MiniMaxH3FantasticRefModTextEncode`).
* MiniMax H3 video + audio VAE for anchors, decoding and stitching.

## Installation and rollback

The fork keeps the upstream pack directory name and every upstream node ID, so
it **replaces** the upstream pack in `custom_nodes` — do not install both at
once. Duplicate node registrations are ambiguous.

```sh
# install (replace the upstream checkout)
mv custom_nodes/herrgotts-h3-infinite-continuation-suite /path/to/rollback
git clone -b feat/external-conditioning \
  https://github.com/dewwwey/Herrgotts-H3-Infinite-Continuation-Suite.git \
  custom_nodes/herrgotts-h3-infinite-continuation-suite
# restart ComfyUI
```

Rollback: remove the new directory and move the preserved upstream checkout
back into place, then restart ComfyUI. Existing workflows keep working in
either direction because no node ID, socket name or socket order was changed.

## Tests

```sh
python -m pytest -q            # 176 tests, including tests/test_external_conditioning.py
python tools/validate_split_workflows.py --base http://127.0.0.1:8188
```

`tests/test_external_conditioning.py` covers registration, the socket contract,
geometry/duration validation, the conditioning contract (multi-entry +
scheduled, immutability, identity, tag validation, keyframe/frame-count
conflicts, unknown-field preservation), temporal anchor placement, and
equivalence of the Continue node's duration plan, source slice, target AV shape,
protected values, masks, context counts and ignored-tail counts against
upstream `H3ContinuousContinueV14` for both audio-tail modes and zero/nonzero
feather. The equivalence tests use stub CLIP/VAE objects — they are structural,
not inference.

`tests/test_external_conditioning_examples.py` covers the three split example
graphs: link integrity, which node feeds the sampler/guider, the saved clip
indices, the head-context and handover wiring, the last-frame save/load path,
and that the unused Fantastic latent output feeds nothing.

`tests/test_example_workflows.py` additionally verifies that every registered
node ID is documented in `node_list.json` (including the external-conditioning
module) and that the v1.4 planner still lives in `nodes.py`.

`tests/test_external_conditioning.py` also covers the `last_frame` output of
the Stitch Output nodes: Full mode returns the literal final frame, Stitch
Ready returns the exact phase-aligned join frame (not the trimmed tail), and
Final Clip keeps the landing after the head trim.

## Real-run acceptance record

The jobs were executed against the live ComfyUI 0.38.0 instance on an
RTX 3090 (704x704, 8 steps, one character RefMod plus one media picture):

| Check | Result |
| --- | --- |
| Job 1 (Start, 5 s) | **pass** — 138 s, `mode=T2VA`, `124 H3 frames`, `conditioning entries 1`, `inherited refs 2 (unchanged)`, `text encoder passes added 0` |
| Job 1 render | `output/video/H3_External_Conditioning_01_0000N_.mp4` |
| Job 1 last frame | `output/h3_external_conditioning/handover_last_frame_00001_.png` — post-trim join frame (source frame 106 of 124; 17-frame unusable tail excluded) |
| Clip 1 saved state | `output/h3_external_conditioning/shot_00001.safetensors`, video `[1,24,37,44,44]`, audio `[1,32,2,207]`, `frame_count 124`, `handover_json` present (`handover_end_frame 106`, `landing_tail_frames 17`, `masked_av_source_policy safe_handover_window`, `no_lock_fallback_applied true`) |
| Job 2 (Continue, 5 s) | **pass** — 177 s; protected source 68..106 of 124 (39 protected frames, unusable tail excluded); `duration=net_new_content -> total 158f`; Fantastic reference map shows `<Picture 1> = picture 1 (media)` (the loaded last frame) plus the FaceCrop RefMod |
| Job 3 (Stitch) | **pass** — 51 s; "saved-chain seamless stitch complete, clips 1-2, 226 frames @ 24fps", clip 2 head trim 39, join crossfade 4f/15ms |
| Continuity scan | **pass** — ffmpeg scene-score scan of the final stitch finds **no hard cut** (the pre-fix run cut to an unrelated indoor scene at t≈5.17 s); exported `last_frame` vs stitched seam frame differ by ~1% mean channel value (codec noise) |

The first job-2 attempt failed and exposed a real pre-existing bug: in this
ComfyUI build `folder_paths.get_save_image_path()` writes under the
`--base-directory` output root while `folder_paths.get_output_directory()`
returns `ComfyUI/output`, so **Load AV Latent could not find what Save AV Latent
had just written**. `_resolve_latent_path()` now searches every candidate output
root (including the location `get_save_image_path` actually uses) instead of
assuming the two agree.

The first successful two-job run then exposed the scene-continuity gap: the
stitch was seamless at the join, but the continuation body cut to an unrelated
scene ~0.7 s in, because the text encoder had no picture of the ongoing scene.
The `last_frame` picture handover above was added and verified in a second
full run.

