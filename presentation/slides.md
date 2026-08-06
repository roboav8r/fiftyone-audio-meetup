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

### Curating, Searching, and Evaluating Audio Datasets in FiftyOne

**John Duncan, Ph.D.**

Machine Learning Engineer, Customer Success

<p class="wordmark">voxel51</p>

Note:
Thank the organizers.
Intro: Customer Success MLE At Voxel51, work with enterprise clients
Prior: Robot Audition during doctoral research
Hook: Show you ways that you can use FiftyOne for your Audio ML tasks and research

---

## Agenda

- Intro material
- A FiftyOne Audio primer
- Task 1: Representing audio
- Task 2: Curation and Audio similarity search
- Task 3: Audio captioning
- Task 4: Audio moment retrieval
- Summary & Conclusions

Note:
Keep this to 30s, just orient the room.

---

## Ground Rules

- Materials available online:
  - [FiftyOne Audio Toolkit](https://github.com/roboav8r/fiftyone-audio-toolkit): custom audio operators, panels, renderers
  - [FiftyOne Audio Meetup](https://github.com/roboav8r/fiftyone-audio-meetup/tree/main): this presentation, helper scripts (dataloader, inference, evaluation)
- This is a tooling/data platform discussion, not research-focused
- Hands-on: will transition between this slide deck and live demo

---

## What is FiftyOne?

- FiftyOne is an open-source toolkit to visualize, curate, and evaluate datasets
- Common workflows: **curate → annotate → infer → evaluate**
- Originally designed for images & video
  - Multimodality and Physical AI is a new addition
  - Audio is not natively supported (yet!); many features are custom extensions of core FiftyOne

Note:
Short, assume the crowd knows Voxel51 already. Just orient anyone new.

----

## FiftyOne Data Model, for Audio

- **Sample** —  a single audio clip
- **Fields** — additional metadata on a sample: labels, captions, evaluation scores
- **Dataset** — samples organized around one task: classification, captioning, moment retrieval
- **View** — a smaller, focused subset of a dataset (splits, folds)

- **SDK/Operators/Plugins** — ways of extending core FiftyOne functionality

Note:
Sample: Will discuss tradeoffs in Task 1
Fields: Classification and TemporalDetection are the two label types we'll see today.

SDK: how I added audio support customization; script -> operator -> plugin

---

<!-- .slide: data-background-color="#1c1e22" -->

## Coming up

- <span class="agenda-dim">Intro material</span>
- <span class="agenda-dim">A FiftyOne Audio primer</span>
- <span class="agenda-current">Task 1: Representing audio</span>
- <span class="agenda-dim">Task 2: Curation and Audio similarity search</span>
- <span class="agenda-dim">Task 3: Audio captioning</span>
- <span class="agenda-dim">Task 4: Audio moment retrieval</span>
- <span class="agenda-dim">Summary & Conclusions</span>

---

<!-- .slide: data-background-color="#1c1e22" -->

<p class="eyebrow">Task 1</p>

# Representing audio

- FiftyOne has **no native audio media type**
- a `.wav`/`.mp3` sample is `media_type = "unknown"`
- two methods to represent audio, each with advantages/disadvantages

----

## Method 1: spectrogram sample renderer (native `.wav`)

- custom `SampleRenderer` plugin (FiftyOne Enterprise 2.18+ only)
- draws a spectrogram + audio player **in the grid**
- client-side STFT (Web Audio), precomputed PNG as fallback
- works for `wav` / `mp3` / `flac` / `ogg`

> Demo: `esc-50-wav` spectrograms render live in the grid, clips play in the modal.

----

## Method 2 — convert audio to spectrogram video (.mp4)

- render the full clip as a **spectrogram video** with audio track
- inherits video methods (Relevant in Task 4: Audio Moment Retrieval)

> Demo: `esc-50` videos in the grid, videos play in the modal.

----

## ESC-50 has a Classification field

- `category` — the 50 fine-grained sound classes, as `fo.Classification`
- `major_category` — 5 coarse groups (Animals, Natural soundscapes, Human non-speech, Interior/domestic, Exterior/urban), also `fo.Classification`
- a `fo.Classification` label = a single string `label` + optional `confidence` — the simplest FiftyOne label type

Note:
Sets up Task 2: once a field is a `fo.Classification`, you can filter/sort/subset a dataset on it directly.

---

<!-- .slide: data-background-color="#1c1e22" -->

## Coming up

- <span class="agenda-dim">Intro material</span>
- <span class="agenda-dim">A FiftyOne Audio primer</span>
- <span class="agenda-dim">Task 1: Representing audio</span>
- <span class="agenda-current">Task 2: Curation and Audio similarity search</span>
- <span class="agenda-dim">Task 3: Audio captioning</span>
- <span class="agenda-dim">Task 4: Audio moment retrieval</span>
- <span class="agenda-dim">Summary & Conclusions</span>

Note:
Point out this is a complexity ramp, not four unrelated demos — CLAP/MS-CLAP (text↔audio) threads through all of them.

---

<!-- .slide: data-background-color="#1c1e22" -->

<p class="eyebrow">Task 2</p>

# Curation and Audio similarity search

### ESC-50

Note:
Orient: Next, will build upon the basic dataset and classification concept

we'll look at ESC-50 through fields → sorting/subsets → the Classification label, then curate with the embeddings-search panel.

----

## Audio Embeddings Search panel

- Embeddings provide a means of identifying clusters and outliers
- They can be computed via SDK script or in-app via delegating to a runner
  - CLAP **or MS-CLAP** enable text-query sim search

> Demo: open embeddings panel, click into a cluster, run a text query, mine a
> below-threshold outlier, tag it.

Note:
This is the "curate" beat — show, don't just tell. Multiple embeddings backends are available; CLAP and MS-CLAP will disagree on some borderline clips, which is itself a good talking point.

---

<!-- .slide: data-background-color="#1c1e22" -->

## Coming up

- <span class="agenda-dim">Intro material</span>
- <span class="agenda-dim">A FiftyOne Audio primer</span>
- <span class="agenda-dim">Task 1: Representing audio</span>
- <span class="agenda-dim">Task 2: Curation and Audio similarity search</span>
- <span class="agenda-current">Task 3: Audio captioning</span>
- <span class="agenda-dim">Task 4: Audio moment retrieval</span>
- <span class="agenda-dim">Summary & Conclusions</span>

---

<!-- .slide: data-background-color="#1c1e22" -->

<p class="eyebrow">Task 3</p>

# Audio captioning - Clotho Dataset

- **Load**: Clotho dataset via dataloader
  - 15-30 second clips of varying complexity
- `captions` field: 5 human descriptions (text)
- `ground_truth` field: primary reference caption (caption_1) from `captions` (label)

----

## Captioning Workflow

- **infer**: CoNeTTE model script → `predictions`
- **evaluate**: compare `ground_truth` and `predictions` via `aac_metrics` script
  - `aac_metrics` → SPIDEr / FENSE / CIDEr-D / SPICE

> Corpus SPIDEr 0.353 / FENSE 0.518 — in line with published CoNeTTE numbers.

<small>References: Clotho ([Drossos et al., ICASSP 2020](https://arxiv.org/abs/1910.09387))
· CoNeTTE ([Labbé et al., TASLP 2024](https://arxiv.org/abs/2309.00454))
· aac-metrics/FENSE ([Zhou et al.](https://arxiv.org/abs/2110.04684),
[toolkit](https://github.com/Labbeti/aac-metrics))</small>

Note:
Demo: sort ascending by SPIDEr to surface the worst captions, open one in the modal, read GT vs. predicted aloud.
Metric cheat sheet: SPIDEr = avg(CIDEr-D, SPICE) — n-gram + scene-graph overlap vs. references; was DCASE's official AAC metric through 2022.
SPIDEr-FL = SPIDEr penalized by a fluency-error probability (repeated n-grams, incomplete sentences, etc.) — DCASE's official metric since 2023.
FENSE = SentenceBERT similarity + the same fluency penalty — correlates much better with human judgment (~76% on Clotho-Eval) than the n-gram metrics.

---

<!-- .slide: data-background-color="#1c1e22" -->

## Coming up

- <span class="agenda-dim">Intro material</span>
- <span class="agenda-dim">A FiftyOne Audio primer</span>
- <span class="agenda-dim">Task 1: Representing audio</span>
- <span class="agenda-dim">Task 2: Curation and Audio similarity search</span>
- <span class="agenda-dim">Task 3: Audio captioning</span>
- <span class="agenda-current">Task 4: Audio moment retrieval</span>
- <span class="agenda-dim">Summary & Conclusions</span>

---

<!-- .slide: data-background-color="#1c1e22" -->

<p class="eyebrow">Task 4</p>

# Audio moment retrieval

- DCASE 2026 Task 6: given a long clip + a text query, return the matching `[start, end]` moment(s)

----

## Moment retrieval — Clotho-Moment

- **Load**: Clotho-Moment spectrogram video via dataloader
  - native timeline + audio + `TemporalDetections`
<!-- - GT moments vs. a text query → predicted moment -->
- **Compute**: sliding-window CLAP/MS-CLAP embeddings script (10s window, 1s hop)
  per clip; find the window closest to the query at each timestamp
- **Evaluate**: Use native temporal eval: **ActivityNet mAP / recall@IoU**

| Backend | R1@0.5 | R1@0.7 | mAP |
|---|---|---|---|
| CLAP | 0.739 | 0.446 | 0.397 |
| MS-CLAP | 0.731 | 0.385 | 0.351 |

> 5,741 valid-split clips, 4,918 ground-truth queries — full methodology in
> `dcase26-task6-amr/retrieval/README.md`.

<small>Reference: Clotho-Moment — [Munakata et al., "Language-based Audio
Moment Retrieval," ICASSP 2025](https://arxiv.org/abs/2409.15672)</small>

Note:
Demo: open a clip, show the GT moment on the timeline, run a text query, show the predicted moment land close to it.
This is the highlight of the talk — slow down here if the room is engaged.
`view.evaluate_detections("predictions", gt_field="ground_truth", method="activitynet", compute_mAP=True).mAP()`
Metric cheat sheet: R1@0.5/R1@0.7 = did the top-1 predicted moment overlap the ground truth by at least that IoU? (fraction of queries). mAP = precision-recall averaged across IoU thresholds, computed per-query (classwise) so a prediction only counts against its own query's ground truth.
Method note: this is a simple, zero-training heuristic, not a learned detector — cosine-similarity between the text query embedding and every window embedding of that clip, smoothed (3-window moving average), take the peak, then expand outward while the score stays within a small margin (0.05) of the peak to get the predicted [start, end]. It's meant to be illustrative of what you can do with off-the-shelf embeddings and no training, not a leaderboard contender: the official DCASE leaderboard (scored on the real-world CASTELLA dataset, not Clotho-Moment) is dominated by trained DETR-style temporal detectors and ensembles — apples-to-oranges versus this threshold heuristic on the easier synthetic Clotho-Moment set.

---

<!-- .slide: data-background-color="#1c1e22" -->

## Coming up

- <span class="agenda-dim">Intro material</span>
- <span class="agenda-dim">A FiftyOne Audio primer</span>
- <span class="agenda-dim">Task 1: Representing audio</span>
- <span class="agenda-dim">Task 2: Curation and Audio similarity search</span>
- <span class="agenda-dim">Task 3: Audio captioning</span>
- <span class="agenda-dim">Task 4: Audio moment retrieval</span>
- <span class="agenda-current">Summary & Conclusions</span>

---

<!-- .slide: data-background-color="#1c1e22" -->

## Takeaways

- Core FiftyOne data types and workflows extend to Audio
- Representation: **custom renderer** + **spectrogram video**
- Embeddings search: **CLAP and MS-CLAP** sim search, positive/negative sample retrieval using Audio search panel
- Captioning: Inspect high and low quality predictions; surfaced data quality issues
- Moment Retrieval: Timeline tools and eval panel surface failure modes
- <span class="highlight-item">A range of Audio ML tasks are possible within FiftyOne</span>
- <span class="contact-highlight">Further collaboration: john@voxel51.com</span>

<p class="wordmark">voxel51</p>

### Thanks — questions?

Note:
Leave this slide up for Q&A. Have the FiftyOne app tab ready to jump back into for follow-up questions.
