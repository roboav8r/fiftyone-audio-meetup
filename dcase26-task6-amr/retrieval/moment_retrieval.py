#!/usr/bin/env python
"""Track B moment retrieval: query-similarity threshold over CLAP/MS-CLAP
sliding-window embeddings.

For each (clip, query) pair with a ground-truth moment (per the manifest's
parallel captions/moments_sec/qids lists), score every window of that SAME
clip against the query's text embedding, smooth, threshold relative to the
query's own peak similarity (robust to CLAP-vs-MS-CLAP score-scale
differences without a separate tuning pass), and merge the contiguous
above-threshold run around the peak into one predicted [start, end] --
a single top-1 prediction per query, matching the R1/mAP retrieval framing.

Reads window-embedding sidecars from retrieval/clap_windows.py's output
(``scratch/clap/<backend>/<split>/<vid>.{npy,json}``).

Usage:
    python retrieval/moment_retrieval.py --split valid --backend clap
    python retrieval/moment_retrieval.py --split valid --backend msclap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="valid")
    p.add_argument("--backend", default="clap", choices=["clap", "msclap"])
    p.add_argument("--manifest", default=str(PROJECT_ROOT / "scratch" / "manifest.jsonl"))
    p.add_argument("--sidecar-dir", default=None,
                   help="defaults to scratch/clap/<backend>/<split>/")
    p.add_argument("--margin", type=float, default=0.05,
                   help="threshold = peak_similarity - margin")
    p.add_argument("--smooth-k", type=int, default=3,
                   help="moving-average window over per-window similarity scores")
    p.add_argument("--out", default=None,
                   help="defaults to scratch/predictions_<backend>.jsonl")
    return p.parse_args()


def read_positive_rows(manifest: Path, split: str):
    rows = []
    with manifest.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split or row.get("num_moments", 0) == 0:
                continue
            rows.append(row)
    return rows


def load_embed_texts(backend: str):
    if backend == "clap":
        from clap.clap import embed_texts
    elif backend == "msclap":
        from clap.msclap import embed_texts
    else:
        raise ValueError(backend)
    return embed_texts


def predict_window(sims: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                    margin: float, smooth_k: int):
    """Smooth similarity scores, take the peak, and expand outward while
    scores stay within `margin` of the peak. Returns (start, end, confidence)."""
    if smooth_k > 1 and len(sims) >= smooth_k:
        kernel = np.ones(smooth_k) / smooth_k
        smoothed = np.convolve(sims, kernel, mode="same")
    else:
        smoothed = sims

    peak_i = int(smoothed.argmax())
    threshold = smoothed[peak_i] - margin

    lo = peak_i
    while lo > 0 and smoothed[lo - 1] >= threshold:
        lo -= 1
    hi = peak_i
    while hi < len(smoothed) - 1 and smoothed[hi + 1] >= threshold:
        hi += 1

    return float(starts[lo]), float(ends[hi]), float(smoothed[peak_i])


def main() -> None:
    args = parse_args()
    sidecar_dir = Path(args.sidecar_dir) if args.sidecar_dir else (
        PROJECT_ROOT / "scratch" / "clap" / args.backend / args.split
    )
    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "scratch" / f"predictions_{args.backend}.jsonl"
    )
    embed_texts = load_embed_texts(args.backend)

    rows = read_positive_rows(Path(args.manifest), args.split)
    print(f"{len(rows)} positive clips for split={args.split!r}")

    # Flatten to one entry per (vid, qid, caption) -- a clip can have multiple.
    flat = []
    for r in rows:
        for caption, qid in zip(r["captions"], r["qids"]):
            flat.append({"vid": r["vid"], "qid": qid, "caption": caption})
    print(f"{len(flat)} (clip, query) pairs; embedding all query texts ...")

    query_embs = embed_texts([f["caption"] for f in flat])
    for f, emb in zip(flat, query_embs):
        f["query_emb"] = emb

    by_vid = {}
    for f in flat:
        by_vid.setdefault(f["vid"], []).append(f)

    predictions = []
    missing_sidecars = []
    n_done = 0
    for vid, queries in by_vid.items():
        npy = sidecar_dir / f"{vid}.npy"
        meta = sidecar_dir / f"{vid}.json"
        if not npy.is_file() or not meta.is_file():
            missing_sidecars.append(vid)
            continue
        embs = np.load(npy)
        d = json.loads(meta.read_text())
        starts = np.asarray(d["starts"])
        ends = np.asarray(d["ends"])

        for q in queries:
            sims = embs @ q["query_emb"]
            s, e, conf = predict_window(sims, starts, ends, args.margin, args.smooth_k)
            predictions.append({
                "vid": vid, "qid": q["qid"], "start": s, "end": e, "confidence": conf,
            })
        n_done += 1
        if n_done % 200 == 0:
            print(f"  scored {n_done}/{len(by_vid)} clips")

    if missing_sidecars:
        print(f"\n{len(missing_sidecars)} clips missing sidecars under {sidecar_dir} "
              f"(e.g. {missing_sidecars[0]}) -- run clap_windows.py first. Skipped.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"\nWrote {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    main()
