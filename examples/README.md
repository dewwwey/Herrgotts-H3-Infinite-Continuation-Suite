# Example workflows — v1.4

The v1.4 examples use **native Masked AV continuation** and require a current ComfyUI build containing PR #15375 H3 AV-mask support.

v1.4 uses the validated video architecture: Auto Handover finds a freeze/brightness-safe visual endpoint, snaps it to the exact Masked-AV video boundary, `Stitch Ready` ends there, and the next protected video context ends at that same source point.

Recommended v1.4 defaults:

```text
Duration Mode: Net New Content (default)
Audio Tail Carryover: Full Previous Tail (default)
Audio feather: 0 ticks
Safe Tail Bridge: 0
Luminance Match: Off
```

`Net New Content` samples a longer total latent so the requested Continue duration approximately describes newly generated visible video after the protected head. It therefore increases sampling time/memory compared with `Total Generation`.

`Full Previous Tail` keeps the safe video cut but allows valid original audio to remain protected beyond that cut. This is intended for dialogue that continues into video frames discarded because of freeze/brightness landing. `Match Video Handover` reproduces video-matched audio behavior.

## 1. `Herrgotts_H3_Infinite_v1.4_01_Start.json`

Creates Clip 1 with flexible T2VA/I2VA/L2VA/FL2VA conditioning. First/Last Frames remain optional. The example uses First + Last because repeated endpoints are the recommended quality-reset workflow. Auto Handover analyzes the rendered result and the full AV latent is saved with its handover metadata.

## 2. `Herrgotts_H3_Infinite_v1.4_02_Continue.json`

Loads the previous full AV latent **and its handover metadata**. The Continue node protects the safe 39-frame video history ending before the unusable landing tail. Audio starts at the same source position but, by default, stays protected through the previous clip's actual end.

Picture mapping keeps v1.3 semantics:

```text
Previous masked AV context = not a Picture
Last Frame = Picture 1 (when connected)
Qwen Reference 1 = Picture 2
Qwen Reference 2 = Picture 3
...
```

Save / Load clip indices remain manual and predictable.

## 3. `Herrgotts_H3_Infinite_v1.4_03_3Clip_Showcase_AutoStitch.json`

Complete one-queue demonstration:

```text
Clip 1 Start
→ Clip 2 Native Masked AV Continue
→ Clip 3 Native Masked AV Continue
→ v1.4 Seamless AV Joins
→ final video
```

The stable shared video seam is used throughout. Later-frame recovery is intentionally not used because live testing showed visible motion/alignment errors and no net-new content benefit. Extra protected audio is already embedded in each next latent and remains naturally after the normal duplicate-head trim.

## 4. `Herrgotts_H3_Infinite_v1.4_04_Stitch_Saved_Chain.json`

For long projects generated clip-by-clip. It decodes one saved AV latent at a time, uses the saved safe video boundary/head metadata, and preserves extended audio tails automatically. Peak memory stays tied roughly to one decoded clip rather than the whole chain. Older v1.3 and experimental v1.4 metadata remain supported for compatibility.

## Dialogue testing

For speech, keep `audio_feather_ticks = 0`. A useful A/B test is a clip whose video becomes unusable before a spoken word has fully finished:

- `Full Previous Tail`: should preserve the already-generated word ending in Clip 2 while video begins generating after the safe visual seam.
- `Match Video Handover`: intentionally cuts protected audio at the visual seam and reproduces the validated video path behavior.

If the original source clip itself ends before the word is complete, v1.4 cannot preserve audio that does not exist.

---

# External conditioning examples (fork feature branch)

These three graphs replace the in-graph text encoder with
**Fantastic RefMod Text Encode → H3 Start/Continue — External Conditioning**.
They are the same workflow split into three separately runnable jobs so a long
segment chain never depends on loop-carried in-memory state.

Prerequisites: Fantastic installed, a current ComfyUI with native MiniMax H3
Masked-AV support (PR #15375), and the MiniMax H3 video + audio VAE.

## A. `Herrgotts_H3_Infinite_v1.5_ExternalConditioning_01_Start.json`

**Job 1 — segment 1 (5 s).**

```text
RefMod Stack + Media Loader + Prompt Builder
→ Fantastic RefMod Text Encode        (the only text-encode pass)
→ H3 Start — External Conditioning    (AV latent, optional First/Last temporal anchors)
→ sampler → VAE decode (video + audio)
→ Auto Handover v1.4
→ Save AV Latent, clip 1              (h3_external_conditioning/shot_00001.safetensors)
→ Output / Stitch v1.4 (Stitch Ready) → Save Video
                                      └→ last_frame → SaveImage
                                         (h3_external_conditioning/handover_last_frame_00001_.png)
```

Fantastic's own `latent` output is **not** connected: the start node decides the
generation latent from its own `duration`. Keep Fantastic's `width`/`height` at
the real generation geometry; its `length` only trims ordinary media references.

The graph renders the 5 s body with `Stitch Ready`, which removes the unusable
freeze/brightness tail. The **full** latent (including that tail) is what gets
saved, because the next segment's handover metadata points at the safe endpoint
inside it.

`Stitch Ready`'s `last_frame` output is the **exact post-trim join frame** —
the last frame this clip contributes to a stitched chain. Job 1 saves it to
`output/h3_external_conditioning/handover_last_frame_00001_.png`. `SaveImage`
appends a counter on every run: delete stale `handover_last_frame_*.png` before
re-running job 1 so job 2 keeps loading the fresh frame.

## B. `Herrgotts_H3_Infinite_v1.5_ExternalConditioning_02_Continue.json`

**Job 2 — segment 2 (5 s).**

```text
Load AV Latent, clip 1  → latent + handover
Media Loader            → <Picture 1> = job 1's handover_last_frame (… [output] path)
Fantastic RefMod Text Encode
→ H3 Continue — External Conditioning (previous_latent + handover)
→ sampler → VAE decode (video + audio)
→ Auto Handover v1.4
→ Save AV Latent, clip 2
→ Output / Stitch v1.4 (Final Clip) → Save Video
```

`duration = 5.0`, `duration_mode = Net New Content`, `masked_context_frames = 39`
and `audio_tail_carryover = Full Previous Tail` are the shipped defaults: the
segment generates roughly 5 s of new visible body on top of the 39 protected
frames (~1.625 s of reused, already-correct history).

The protected latent head anchors only the first ~1.6 s; the rest of the body
is generated from conditioning. That is why the Media Loader swaps the source's
first-frame picture for job 1's saved last frame: `<Picture 1>` becomes the
exact frame [Shot 2] continues from, which keeps the whole segment in the same
scene. The segment prompt matches: the truck in that starts in [Shot 1]
continues without a cut until it stops on the close-up, instead of cutting to a
new framing.

## C. `Herrgotts_H3_Infinite_v1.5_ExternalConditioning_03_Stitch.json`

**Job 3 — final stitch only.**

```text
Stitch Saved Chain v1.4   (reads shot_00001 + shot_00002 from disk)
→ video/H3_External_Conditioning_final_*.mp4
```

The stitch node reads **both** saved latents straight from disk, so it can also
be run on its own after any number of segments. It runs as its own job because
it has no data dependency on the Save AV Latent nodes: inside one prompt it
would execute before clip 2 exists and silently skip. Its `IS_CHANGED`
fingerprints the saved chain, so re-queuing the stitch after the chain
completes re-runs it instead of returning a cached "incomplete" result.

## Handover between the jobs

Everything passing between the jobs is on disk:

| Channel | File / field |
| --- | --- |
| Previous full sampled AV latent | `h3_external_conditioning/shot_00001.safetensors` (`video`, `audio` tensors) |
| Freeze-safe handover metadata | the same file's `handover_json` safetensors metadata |
| Reused head length | the same file's `head_context_frames` metadata |
| Previous scene picture | `h3_external_conditioning/handover_last_frame_00001_.png` → Media Loader `[output]` path → `<Picture 1>` |

There is no shared in-memory cache between the prompts, and re-running a
completed job overwrites only its own fixed clip index.

## API equivalents

`workflows/job1_start_api.json`, `workflows/job2_continue_api.json` and
`workflows/job3_stitch_api.json` are the same graphs in ComfyUI API format for
headless runs:

```sh
python tools/run_api_workflow.py workflows/job1_start_api.json
python tools/run_api_workflow.py workflows/job2_continue_api.json
python tools/run_api_workflow.py workflows/job3_stitch_api.json
```
