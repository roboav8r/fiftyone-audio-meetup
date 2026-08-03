# DCASE 2026 Task 6 — Audio Moment Retrieval from Long Audio

Stage the **Clotho-Moment** development set into a cloud bucket and browse it
on a FiftyOne deployment with the
[`@roboav8r/fiftyone-audio-toolkit`](https://github.com/roboav8r/fiftyone-audio-toolkit).

- **Task:** given a long (1-minute) recording and a text query, return the
  `[start, end]` moment(s) that match.
  [Task page](https://dcase.community/challenge2026/task-audio-moment-retrieval-from-long-audio)
- **Baseline (QD-DETR):** https://github.com/awkrail/dcase2026_task6_baseline

## Dataset: Clotho-Moment

HuggingFace `lighthouse-emnlp2024/Clotho-Moment`, WebDataset tars, **~197 GB**:

| split | shards | clips  |
|-------|--------|--------|
| train | 716    | 37,930 |
| valid | 109    |  5,741 |
| test  | 143    |  7,569 |
| **total** | **968** | **51,240** |

Each shard tar holds ~53 clips as `<key>.wav` + `<key>.json`. The JSON is the
**generation recipe**, not the flattened baseline annotation:

```json
{"bg": {"path": "...", "dB": -23.7},
 "fg": [{"qid": 32694, "path": "...", "caption": "A toilet flushes ...",
         "dB": 0.7, "duration": 7.0, "start_time": 41.06}]}
```

Each `fg` event is a **moment**: `query = caption`, window =
`[start_time, start_time + duration]`. We store these per clip as parallel list
fields `captions` / `moments_sec` / `qids`.

## Pipeline

Full dataset does not fit on local disk (152 GB free), so we stage
**shard-at-a-time** (download → extract → upload → prune), keeping only a few GB
resident. Two scripts:

1. **`stage_clotho_moment.py`** (no FiftyOne) — HF tar → local wav → cloud
   bucket → prune. Resumable via `scratch/staged_shards.txt`; per-clip
   metadata appended to `scratch/manifest.jsonl`. Media lands at
   `<gcs-root>/<split>/<vid>.wav`.
2. **`build_clotho_moment_dataset.py`** — `manifest.jsonl` → `fo.Dataset`,
   one sample per clip with the wav filepath.
3. **`loaders/convert_clotho_moment_to_video.py`** (in the repo root's
   `loaders/`) — converts a built split to spectrogram-video so it gets a
   native timeline + `ground_truth` `TemporalDetections`; see Part 3 of the
   top-level README.

### Setup

```bash
cd dcase26-task6-amr
cp .env.example .env    # FOE only -- fill in FIFTYONE_API_URI / FIFTYONE_API_KEY / FIFTYONE_PYPI_TOKEN
mamba env create -f environment.yml && mamba activate fo-dcase26-task6-amr
# (or reuse fo-audio-ai-meetup -- same deps)
# install the FiftyOne SDK for the build step: `pip install fiftyone` for
# OSS, or your org's private-index flow for Enterprise

# gcloud must be authenticated to write to your cloud bucket:
#   ! gcloud auth login
```

### Run

```bash
# smoke test (GATE): 1 valid shard -> ~53 wavs -> tiny dataset
python stage_clotho_moment.py --gcs-root gs://your-bucket/clotho-moment \
    --splits valid --shard-limit 1
python build_clotho_moment_dataset.py --env-file .env \
    --name clotho-moment-smoke --limit 50
# -> open the App, confirm the audio-toolkit spectrogram renders and audio
#    plays (i.e. /media serves the wav to the browser).

# full staging (long; background it), then full build
python stage_clotho_moment.py --gcs-root gs://your-bucket/clotho-moment \
    --splits train valid test
python build_clotho_moment_dataset.py --env-file .env --name clotho-moment
```

## Results

Full pipeline (staging → video conversion → CLAP/MS-CLAP sliding-window
embeddings → Track B retrieval → ActivityNet/R1 eval) run on the Clotho-Moment
`valid` split (5,741 clips, 4,918 ground-truth queries):

| Backend | R1@0.5 | R1@0.7 | mAP (ActivityNet) | queries covered |
|---|---|---|---|---|
| clap | 0.7391 | 0.4455 | 0.3973 | 4918/4918 |
| msclap | 0.7308 | 0.3845 | 0.3513 | 4918/4918 |

Full methodology in [retrieval/README.md](retrieval/README.md); the
`train`/`test` splits aren't yet run through this evaluation pipeline.

Note: the toolkit's built-in embeddings search is **whole-clip CLAP
similarity** (text→clip), which is *not* this Task-6 temporal-grounding
metric -- that's what `retrieval/` computes separately, per-window.

## Out of scope (follow-ups)

- **Evaluation set** (100 clips): only MS-CLAP features are public; raw audio
  needs an email to the organizers.
- **CASTELLA** (1,862 real YouTube clips): captions on `github.com/line/castella`,
  audio via `github.com/h-munakata/CASTELLA-audio` (YouTube scraping).
