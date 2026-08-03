#!/usr/bin/env python
"""Write Track B predicted moments into FiftyOne as TemporalDetections.

Reads a predictions JSONL (from moment_retrieval.py: one row per
{vid, qid, start, end, confidence}) and writes them onto the
`clotho-moment-video` dataset as a `predictions_<backend>` field.

Each detection is labeled `str(qid)` -- the SAME convention as
`ground_truth` (see convert_clotho_moment_to_video.py) -- so that
`evaluate_detections(..., classwise=True)` only matches a prediction to the
ground truth for the SAME query, not any other overlapping moment on the
same clip.

Usage:
    python retrieval/write_predictions.py --env-file .env \\
        --backend clap --predictions scratch/predictions_clap.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

AUDIO_MEETUP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUDIO_MEETUP_ROOT))       # -> _lib.env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="clotho-moment-video")
    p.add_argument("--backend", required=True, choices=["clap", "msclap"])
    p.add_argument("--predictions", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from _lib import env
    env.load(args.env_file)

    import fiftyone as fo

    rows = []
    with open(args.predictions) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"{len(rows)} predictions from {args.predictions}")

    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["vid"]].append(r)

    dataset = fo.load_dataset(args.dataset_name)
    field = f"predictions_{args.backend}"

    vid_to_sample = {s.vid: s for s in dataset}
    missing_vids = set(by_vid) - set(vid_to_sample)
    if missing_vids:
        print(f"{len(missing_vids)} predicted vids not found in {args.dataset_name} "
              f"(e.g. {next(iter(missing_vids))}) -- skipped.")

    field_values = {}
    n_dets = 0
    for vid, preds in by_vid.items():
        sample = vid_to_sample.get(vid)
        if sample is None:
            continue
        dets = []
        for p in preds:
            det = fo.TemporalDetection.from_timestamps(
                [p["start"], p["end"]], sample=sample, label=str(p["qid"]),
                confidence=p["confidence"],
            )
            dets.append(det)
        field_values[sample.id] = fo.TemporalDetections(detections=dets)
        n_dets += len(dets)

    dataset.set_values(field, field_values, key_field="id")
    print(f"\nWrote '{field}' ({n_dets} detections across {len(field_values)} samples).")


if __name__ == "__main__":
    main()
