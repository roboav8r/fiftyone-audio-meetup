#!/usr/bin/env python
"""Evaluate Track B moment-retrieval predictions and report results.

Two metrics, both per-query (qid is globally unique across the manifest):
  - mAP via FiftyOne's native `evaluate_detections(method="activitynet",
    classwise=True, compute_mAP=True)` -- classwise=True is what makes this
    correct here, since it only matches a prediction to the ground truth
    TemporalDetection with the SAME `label` (str(qid)), not any other
    overlapping moment on the same clip.
  - R1@0.5 / R1@0.7 (top-1 recall at an IoU threshold) computed directly
    from the manifest + predictions JSONL -- the retrieval-standard metric
    for DCASE Task 6, which FiftyOne's generic detection-eval framework
    doesn't expose natively.

Usage:
    python retrieval/eval_moment_retrieval.py --env-file .env \\
        --backends clap msclap --split valid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_MEETUP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(AUDIO_MEETUP_ROOT))       # -> _lib.env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="clotho-moment-video")
    p.add_argument("--split", default="valid")
    p.add_argument("--manifest", default=str(PROJECT_ROOT / "scratch" / "manifest.jsonl"))
    p.add_argument("--backends", nargs="+", default=["clap", "msclap"])
    p.add_argument("--predictions-dir", default=str(PROJECT_ROOT / "scratch"))
    p.add_argument("--readme-out", default=str(PROJECT_ROOT / "retrieval" / "README.md"))
    return p.parse_args()


def temporal_iou(a, b) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def gt_windows_by_qid(manifest: Path, split: str):
    by_qid = {}
    with manifest.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            for window, qid in zip(row["moments_sec"], row["qids"]):
                by_qid[qid] = tuple(window)
    return by_qid


def recall_at_iou(gt_by_qid: dict, preds_by_qid: dict, threshold: float) -> float:
    hits = total = 0
    for qid, gt_window in gt_by_qid.items():
        total += 1
        pred = preds_by_qid.get(qid)
        if pred is not None and temporal_iou(gt_window, pred) >= threshold:
            hits += 1
    return hits / total if total else 0.0


def load_predictions(path: Path):
    by_qid = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            by_qid[row["qid"]] = (row["start"], row["end"])
    return by_qid


def main() -> None:
    args = parse_args()
    from _lib import env
    env.load(args.env_file)

    import fiftyone as fo

    gt_by_qid = gt_windows_by_qid(Path(args.manifest), args.split)
    print(f"{len(gt_by_qid)} ground-truth queries in split={args.split!r}")

    dataset = fo.load_dataset(args.dataset_name)

    rows = []
    for backend in args.backends:
        pred_path = Path(args.predictions_dir) / f"predictions_{backend}.jsonl"
        preds_by_qid = load_predictions(pred_path)
        n_covered = sum(1 for q in gt_by_qid if q in preds_by_qid)
        print(f"\n[{backend}] {len(preds_by_qid)} predictions "
              f"({n_covered}/{len(gt_by_qid)} queries covered)")

        r1_05 = recall_at_iou(gt_by_qid, preds_by_qid, 0.5)
        r1_07 = recall_at_iou(gt_by_qid, preds_by_qid, 0.7)

        field = f"predictions_{backend}"
        results = dataset.evaluate_detections(
            field, gt_field="ground_truth", method="activitynet",
            iou=0.5, classwise=True, compute_mAP=True, eval_key=f"eval_{backend}",
        )
        mAP = results.mAP()

        print(f"[{backend}] R1@0.5={r1_05:.4f}  R1@0.7={r1_07:.4f}  mAP={mAP:.4f}")
        rows.append({"backend": backend, "R1@0.5": r1_05, "R1@0.7": r1_07, "mAP": mAP,
                     "n_queries": len(gt_by_qid), "n_covered": n_covered})

    lines = [
        "# Track B (query-similarity threshold) results — Clotho-Moment validation split",
        "",
        f"DCASE'26 Task 6 moment retrieval. `{len(gt_by_qid)}` ground-truth queries, "
        f"validation split. Predictions: relative-peak-threshold merge over "
        f"10s/1s-hop sliding-window embeddings (see `retrieval/moment_retrieval.py`).",
        "",
        "| Backend | R1@0.5 | R1@0.7 | mAP (ActivityNet, classwise) | queries covered |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['backend']} | {r['R1@0.5']:.4f} | {r['R1@0.7']:.4f} | "
            f"{r['mAP']:.4f} | {r['n_covered']}/{r['n_queries']} |"
        )
    lines.append("")
    lines.append(
        "mAP uses FiftyOne's native `evaluate_detections(method=\"activitynet\", "
        "classwise=True, compute_mAP=True)` over `predictions_<backend>` vs "
        "`ground_truth` on the `clotho-moment-video` dataset; "
        "`classwise=True` + per-query `str(qid)` labels ensure a prediction only "
        "matches the ground truth for the SAME query."
    )
    Path(args.readme_out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote results table to {args.readme_out}")


if __name__ == "__main__":
    main()
