# FiftyOne Audio Meetup Demo

Dataloaders, curation/eval scripts, and the talk deck for a demo of
FiftyOne's **curate → annotate → infer → evaluate** workflow applied to
*audio*. See [presentation/slides.md](presentation/slides.md) for the deck
(run it with `presentation/`'s own [README](presentation/README.md)).

FiftyOne has no native audio media type (a `.wav` is media_type `unknown`),
so the demo bridges that two ways:
- **short clips** → a custom **spectrogram sample renderer** draws them in
  the grid;
- **long clips** → rendered as **spectrogram video** (audio muxed in) so we
  get the native timeline, audio playback, temporal detections, and
  temporal eval.

**Depends on [`fiftyone-audio-toolkit`](https://github.com/roboav8r/fiftyone-audio-toolkit)**
for both of those: the spectrogram renderer and the CLAP/MS-CLAP
embeddings-search panel + operators. Install it on your deployment first —
this repo only holds the dataloaders, curation/eval scripts, and the deck.

## Prerequisites

- **A FiftyOne installation** — either:
  - **OSS**: `pip install fiftyone`, no `.env` needed. You can run every
    loader/curation/eval script and browse the resulting datasets, but the
    custom spectrogram renderer is Enterprise-only, so short audio clips
    won't render visually in the grid.
  - **FOE** (Enterprise): copy `.env.example` → `.env` and fill in
    `FIFTYONE_API_URI` / `FIFTYONE_API_KEY` / `FIFTYONE_PYPI_TOKEN`, then
    install the SDK per your org's usual private-index flow.
- **`fiftyone-audio-toolkit`** installed on that deployment (see its own
  README) if you want the renderer and embeddings-search panel.
- **This repo's Python env**:
  ```bash
  mamba env create -f environment.yml
  mamba activate fo-audio-ai-meetup
  ```
  `ffmpeg` (with `showspectrum`/`showspectrumpic`) comes from conda-forge in
  the env and is required for the spectrogram videos.

## Part 0: Loading audio datasets

Every `loaders/*.py` script follows the same shape:

```bash
python loaders/<script>.py [--env-file .env] [--max-samples N] \
    [--cloud-prefix gs://your-bucket/<name>]
```

- `--env-file` loads an env file before importing fiftyone (defaults to this
  repo's own `.env`; omit entirely for local OSS).
- `--max-samples` caps how much is built, for quick iteration.
- `--cloud-prefix` points each sample's `filepath` at media you've already
  uploaded to a bucket your deployment can read — required for a hosted
  deployment to serve the media; omit to build with local file paths
  instead.

| Dataset | Loader | Used by |
|---|---|---|
| ESC-50 | `loaders/load_esc50.py` | Part 1 |
| Clotho | `loaders/load_clotho.py` | Part 2 |
| Clotho-Moment | `loaders/load_clotho_moment.py` | Part 3 |

## Part 1: Classification (ESC-50)

2,000 five-second environmental-sound clips, 50 classes.

```bash
python loaders/load_esc50.py --cloud-prefix gs://your-bucket/esc50
```

- **Represent**: the toolkit's spectrogram renderer draws each clip in the
  grid; the `@voxel51/dataset-table-view` panel (vendored in `reference/`)
  browses it as a metadata table.
- **Curate**: compute CLAP embeddings and `compute_visualization` (UMAP) to
  spot clusters/outliers; the toolkit's Audio Embeddings Search panel does
  text-query similarity search to surface mislabeled or off-topic clips.

## Part 2: Captioning (Clotho)

15–30s clips, 5 human reference captions each.

```bash
python loaders/load_clotho.py --split validation \
    --cloud-prefix gs://your-bucket/clotho/validation
```

- **Annotate → infer**: `curation/apply_conette.py` runs CoNeTTE captioning
  and writes `predictions`. Requires a LOCAL copy of the audio (a cloud
  `filepath` can't be read by the model directly) and its own conda env
  (`fo-clotho-caption`, pins `torch==1.13.1`) — see the script's docstring.
- **Evaluate**: `curation/evaluate_captions.py` scores `predictions` against
  `ground_truth` with `aac_metrics` (SPIDEr/FENSE/CIDEr-D/SPICE), in its own
  env (`fo-clotho-eval`) since `aac_metrics` wants a much newer torch than
  CoNeTTE. First run needs `aac-metrics-download` for its external
  resources (spaCy model, METEOR jar, FENSE's sentence-transformer
  checkpoint).
- The vendored `reference/apply_audio_captioning_model.zip` /
  `evaluate_audio_captions.zip` operators do the same job as App-side
  operators, but read `filepath` directly — use the scripts above instead
  if your media lives in the cloud.
- Reference result on the validation split: corpus SPIDEr 0.353 / FENSE
  0.518, in line with published CoNeTTE-on-Clotho numbers.

## Part 3: Moment retrieval (DCASE'26 Task 6 / Clotho-Moment)

Given a long clip and a text query, return the `[start, end]` moment(s)
that match — [task page](https://dcase.community/challenge2026/task-audio-moment-retrieval-from-long-audio).
Two entry points into the same dataset:

- **Quick look** — a small demo-sized build, entirely self-contained:
  ```bash
  python loaders/load_clotho_moment.py --inspect        # confirm the schema first
  python loaders/load_clotho_moment.py --max-samples 100 \
      --cloud-prefix gs://your-bucket/clotho-moment
  ```
  Renders each clip as spectrogram video with `ground_truth`
  `TemporalDetections` on the native timeline.

- **Full pipeline** (reproduce the eval numbers below) — stage the full
  51,240-clip corpus, convert to video, and run CLAP/MS-CLAP sliding-window
  retrieval: see [`dcase26-task6-amr/REPRO.md`](dcase26-task6-amr/REPRO.md)
  (staging + dataset build) and
  [`dcase26-task6-amr/retrieval/README.md`](dcase26-task6-amr/retrieval/README.md)
  (retrieval + eval methodology). Video conversion is
  `loaders/convert_clotho_moment_to_video.py` (shared with the quick-look
  path above).

Results on the `valid` split (5,741 clips, 4,918 ground-truth queries):

| Backend | R1@0.5 | R1@0.7 | mAP (ActivityNet) |
|---|---|---|---|
| CLAP | 0.739 | 0.446 | 0.397 |
| MS-CLAP | 0.731 | 0.385 | 0.351 |

Evaluated natively in FiftyOne:
```python
view.evaluate_detections("predictions", gt_field="ground_truth",
                         method="activitynet", compute_mAP=True).mAP()
```

## Layout

```
lib/audio.py               spectrogram PNG + spectrogram-video helpers
_lib/env.py                 .env loader (used by every script's --env-file flag)
loaders/                   load_esc50.py, load_clotho.py, load_clotho_moment.py,
                            convert_clotho_moment_to_video.py
curation/                   apply_conette.py, evaluate_captions.py
presentation/               reveal-md talk deck (slides.md)
reference/                  vendored plugins (table view, captioning, eval)
dcase26-task6-amr/          full DCASE'26 Task 6 pipeline (own README/REPRO)
scratch/                    gitignored: downloads, generated spectrograms/videos
```
