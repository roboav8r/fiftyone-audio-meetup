#!/usr/bin/env python
"""Build the `clotho-moment` FiftyOne dataset on a deployment from the manifest.

Reads ``scratch/manifest.jsonl`` (written by ``stage_clotho_moment.py``) and
creates one sample per clip whose ``filepath`` is the ``gs://`` wav URI, so the
Enterprise App serves the media and the audio toolkit's spectrogram renderer
draws it. Ground-truth moments are carried as parallel list fields
(``captions`` / ``moments_sec`` / ``qids``) since the media type is audio
(``unknown``), not video.

The deployment env MUST be loaded before importing fiftyone; do that via
``--env-file`` (defaults to this repo's top-level ``.env`` via ``_lib.env``).

Usage:
    # smoke test: 50 clips into a throwaway dataset
    python build_clotho_moment_dataset.py --env-file .env \\
        --name clotho-moment-smoke --limit 50
    # full build
    python build_clotho_moment_dataset.py --env-file .env --name clotho-moment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_MEETUP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(AUDIO_MEETUP_ROOT))       # -> _lib.env

TASK_URL = "https://dcase.community/challenge2026/task-audio-moment-retrieval-from-long-audio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--name", default="clotho-moment")
    p.add_argument("--manifest", default=str(PROJECT_ROOT / "scratch" / "manifest.jsonl"))
    p.add_argument("--splits", nargs="+", default=None,
                   help="only load these splits (default: all present in manifest)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of clips loaded (for smoke tests)")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--overwrite", action="store_true", default=True)
    return p.parse_args()


def iter_rows(manifest: Path, splits, limit):
    seen = set()
    n = 0
    with manifest.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            if splits and row.get("split") not in splits:
                continue
            vid = row.get("vid")
            if vid in seen:  # defensive dedup (crash-resume can duplicate)
                continue
            seen.add(vid)
            yield row
            n += 1
            if limit and n >= limit:
                return


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_file():
        sys.exit(f"manifest not found: {manifest} (run stage_clotho_moment.py first)")

    from _lib import env
    env.load(args.env_file)  # override=True; BEFORE importing fiftyone

    import fiftyone as fo

    print(f"[env] API={__import__('os').environ.get('FIFTYONE_API_URI', '?')}")

    dataset = fo.Dataset(args.name, overwrite=args.overwrite)
    dataset.persistent = True

    def make_sample(row: dict) -> fo.Sample:
        s = fo.Sample(filepath=row["gs_path"], tags=[row["split"]])
        s["split"] = row["split"]
        s["vid"] = row["vid"]
        s["duration"] = row.get("duration")
        s["num_moments"] = row.get("num_moments", 0)
        s["captions"] = row.get("captions", [])
        s["moments_sec"] = row.get("moments_sec", [])
        s["qids"] = row.get("qids", [])
        return s

    batch, total = [], 0
    for row in iter_rows(manifest, args.splits, args.limit):
        batch.append(make_sample(row))
        if len(batch) >= args.batch_size:
            dataset.add_samples(batch)
            total += len(batch)
            print(f"  added {total} samples", flush=True)
            batch = []
    if batch:
        dataset.add_samples(batch)
        total += len(batch)

    dataset.info["source"] = "Clotho-Moment / DCASE 2026 Task 6 (Audio Moment Retrieval)"
    dataset.info["task_url"] = TASK_URL
    dataset.info["hf_dataset"] = "lighthouse-emnlp2024/Clotho-Moment"
    dataset.save()

    print(f"\nDone. Dataset '{args.name}': {total} samples")
    counts = dataset.count_sample_tags()
    print("split counts (tags):", counts)
    print("first sample filepath:", dataset.first().filepath)


if __name__ == "__main__":
    main()
