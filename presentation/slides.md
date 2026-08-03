---
title: Audio + AI Meetup — FiftyOne
theme: theme/voxel51.css
revealOptions:
  transition: 'slide'
  slideNumber: true
  progress: true
  controls: false
  hash: true
---

<!-- .slide: data-background-color="#1c1e22" -->

# Audio Data Meets FiftyOne

### Representing, curating, and evaluating sound

<p class="wordmark">voxel51</p>

Note:
Welcome / who I am. Thank the organizers.
Hook: "why is a visual-data tool talking about audio?"

---

## Agenda

1. FiftyOne, briefly
2. Part 1 — Representing audio
3. Part 2 — Curate & explore
4. Part 3 — Annotate → infer → evaluate
5. Wrap-up + roadmap

Note:
Keep this to 30s, just orient the room.

---

## What is FiftyOne?

- Open-source toolkit to visualize, curate, and evaluate datasets
- One workflow: **curate → annotate → infer → evaluate**
- Built for images & video first — audio is new territory

Note:
Short — this crowd likely knows Voxel51 already. Just orient anyone new.

---

<!-- .slide: data-background-color="#1c1e22" -->

<p class="eyebrow">Part 1</p>

# Representing audio in FiftyOne

- FiftyOne has **no native audio media type**
- a `.wav`/`.mp3` sample is `media_type = "unknown"`
- two bridges make the rest of the talk possible

----

## Bridge 1 — spectrogram sample renderer

- custom `SampleRenderer` plugin (FOE 2.18+)
- draws a spectrogram + audio player **in the grid**
- client-side STFT (Web Audio), precomputed PNG as fallback
- works for `wav` / `mp3` / `flac` / `ogg`

> Demo: open ESC-50 in the grid — spectrograms render live, play a clip in the modal.

----

## Bridge 2 — long audio as spectrogram video

- render the full clip as a **spectrogram video** (audio muxed in)
- inherits, for free:
  - native timeline + audio playback
  - `TemporalDetection` overlays
  - ActivityNet-style temporal evaluation

----

## The dataset ramp (tied together by CLAP)

| Dataset | Task | Labels |
|---|---|---|
| ESC-50 | classification | `fo.Classification` (category) |
| Clotho | captioning | `ground_truth` caption (5 refs) |
| Clotho-Moment | moment retrieval | `TemporalDetections` on a timeline |

Note:
Point out this is a complexity ramp, not three unrelated demos — CLAP (text↔audio) threads through all three.

---

<!-- .slide: data-background-color="#1c1e22" -->

<p class="eyebrow">Part 2</p>

# Curate & explore

### ESC-50

- CLAP embeddings → `compute_visualization` (UMAP)
- text-query similarity search ("children playing")
- tabular view panel for metadata browsing
- goal: surface a mislabeled or outlier clip

Note:
Demo: open the UMAP, click into a cluster, run a text query, land on an odd-one-out clip.
This is the "curate" beat — show, don't just tell.

---

<!-- .slide: data-background-color="#1c1e22" -->

<p class="eyebrow">Part 3</p>

# Annotate → infer → evaluate

----

## Captioning — Clotho

- `ground_truth`: human captions (5 references/clip)
- **infer**: CoNeTTE → `predictions`
- **evaluate**: `aac_metrics` → SPIDEr / FENSE / CIDEr-D / SPICE
- sort ascending by SPIDEr to find the worst captions

> Corpus SPIDEr 0.353 / FENSE 0.518 — in line with published CoNeTTE numbers.

Note:
Demo: sort by SPIDEr, open the worst caption in the modal, read GT vs. predicted aloud.

----

## Moment retrieval — DCASE 2026 Task 6

### Clotho-Moment

- what's new for DCASE'26
- spectrogram video → native timeline + audio + `TemporalDetections`
- GT moments vs. a text query → predicted moment
- native temporal eval: **ActivityNet mAP / recall@IoU**

| Backend | R1@0.5 | R1@0.7 | mAP |
|---|---|---|---|
| CLAP | 0.739 | 0.446 | 0.397 |
| MS-CLAP | 0.731 | 0.385 | 0.351 |

> 5,741 valid-split clips, 4,918 ground-truth queries — full methodology in
> `dcase26-task6-amr/retrieval/README.md`.

Note:
This is the highlight of the talk — slow down here if the room is engaged.
`view.evaluate_detections("predictions", gt_field="ground_truth", method="activitynet", compute_mAP=True).mAP()`

---

## Roadmap

- multichannel waveform panel
- teaser only — come find me after if this is interesting

---

<!-- .slide: data-background-color="#1c1e22" -->

## Takeaways

- the same **curate → annotate → infer → evaluate** loop works on audio
- two bridges unlock it: **custom renderer** + **spectrogram video**
- **CLAP** threads search, curation, and retrieval through the whole ramp

<p class="wordmark">voxel51</p>

### Thanks — questions?

Note:
Leave this slide up for Q&A. Have the FiftyOne app tab ready to jump back into for follow-up questions.
