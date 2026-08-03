#!/usr/bin/env python
"""Stage DCASE 2026 Task 6 (Clotho-Moment) audio from HuggingFace to GCS.

Clotho-Moment ships as WebDataset tars on
``lighthouse-emnlp2024/Clotho-Moment`` (968 shards, ~197 GB, 51,240 one-minute
clips across train/valid/test). The full set does not fit on the local disk, so
this stager works **one shard at a time** and keeps only a few GB resident:

    download shard tar  ->  extract raw wavs + parse json  ->  upload wavs to
    <gcs-root>/<split>/  ->  prune local files

It is **resumable**: shards already recorded in ``scratch/staged_shards.txt``
are skipped, so Ctrl-C + re-run continues where it left off. Per-clip metadata
is appended to ``scratch/manifest.jsonl`` for the separate build step
(``build_clotho_moment_dataset.py``).

Per-clip tar schema (confirmed by inspecting valid-000):
    <key>.wav   raw 60s recording
    <key>.json  generation recipe: {"bg": {...}, "fg": [ {qid, path, caption,
                dB, duration, start_time}, ... ]}
Each ``fg`` event is a moment: query = ``caption``, window =
``[start_time, start_time + duration]``.

No FiftyOne import here — this runs in the plain staging env.

Usage:
    # smoke test: one valid shard
    python stage_clotho_moment.py --gcs-root gs://your-bucket/clotho-moment \\
        --splits valid --shard-limit 1
    # full run (background it):
    python stage_clotho_moment.py --gcs-root gs://your-bucket/clotho-moment \\
        --splits train valid test
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

# Prevent indefinite hangs on a stalled (esp. unauthenticated / rate-limited) HF
# socket: bound each request so a dead connection raises instead of sleeping.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")

HF_DATASET = "lighthouse-emnlp2024/Clotho-Moment"
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
                   help="which splits to stage (default: all)")
    p.add_argument("--gcs-root", required=True,
                   help="GCS prefix for wavs, e.g. gs://your-bucket/clotho-moment")
    p.add_argument("--scratch", default=str(Path(__file__).resolve().parent / "scratch"),
                   help="working dir for tars/wavs/manifest/checkpoint")
    p.add_argument("--shard-limit", type=int, default=None,
                   help="max shards to process THIS run (across selected splits)")
    p.add_argument("--keep-local", action="store_true",
                   help="do not delete tars/wavs after upload (uses lots of disk)")
    p.add_argument("--no-upload", action="store_true",
                   help="extract + manifest only; skip GCS upload and pruning")
    p.add_argument("--dry-run", action="store_true",
                   help="list the shards that would be processed and exit")
    p.add_argument("--inspect", action="store_true",
                   help="download the first selected shard, dump its layout, exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shard enumeration + checkpoint
# ---------------------------------------------------------------------------


def list_shards(splits) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(HF_DATASET, repo_type="dataset")
    tars = sorted(f for f in files if f.endswith(".tar"))
    return [t for t in tars if t.split("/", 1)[0] in splits]


def load_done(checkpoint: Path) -> set[str]:
    if not checkpoint.is_file():
        return set()
    return {ln.strip() for ln in checkpoint.read_text().splitlines() if ln.strip()}


# ---------------------------------------------------------------------------
# Annotation parsing (Clotho-Moment generation recipe -> moments)
# ---------------------------------------------------------------------------


def parse_moments(obj: dict) -> tuple[list[str], list[list[float]], list[int]]:
    """Return (captions, windows[[s,e]], qids) from a per-clip json recipe."""
    captions, windows, qids = [], [], []
    for ev in obj.get("fg", []) or []:
        try:
            start = float(ev["start_time"])
            dur = float(ev.get("duration", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        captions.append(str(ev.get("caption", "")).strip())
        windows.append([round(start, 3), round(start + dur, 3)])
        qids.append(ev.get("qid"))
    return captions, windows, qids


def wav_duration(raw: bytes) -> float | None:
    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(raw))
        return round(info.frames / float(info.samplerate), 3)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------


def gsutil_cp(files: list[Path], dest_prefix: str) -> None:
    """Copy a batch of local files into a GCS prefix (dir). Uses `gsutil cp -I`
    reading the source list from stdin so file count / spaces are not a problem.
    """
    if not files:
        return
    payload = "\n".join(str(f) for f in files) + "\n"
    subprocess.run(
        ["gsutil", "-m", "cp", "-I", dest_prefix.rstrip("/") + "/"],
        input=payload, text=True, check=True,
    )


# ---------------------------------------------------------------------------


def process_shard(shard: str, args, seen_vids: set[str], manifest_fh) -> int:
    """Download, extract, upload, prune one shard. Returns #clips staged."""
    from huggingface_hub import hf_hub_download

    split = shard.split("/", 1)[0]
    scratch = Path(args.scratch)
    tars_dir = scratch / "tars"
    wav_dir = scratch / "wav" / split
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{shard}] downloading ...", flush=True)
    tar_path = None
    for attempt in range(1, 6):
        try:
            tar_path = Path(hf_hub_download(
                HF_DATASET, filename=shard, repo_type="dataset",
                local_dir=str(tars_dir), token=os.environ.get("HF_TOKEN")))
            break
        except Exception as ex:  # noqa: BLE001 — network stalls/rate-limits
            wait = min(60, 5 * attempt)
            print(f"[{shard}] download attempt {attempt} failed ({type(ex).__name__}: "
                  f"{str(ex)[:120]}); retrying in {wait}s", file=sys.stderr, flush=True)
            # drop any stale .lock / .incomplete so the retry starts clean
            for junk in (tars_dir / ".cache").rglob("*"):
                if junk.suffix in (".lock", ".incomplete"):
                    junk.unlink(missing_ok=True)
            time.sleep(wait)
    if tar_path is None:
        raise RuntimeError(f"{shard}: download failed after retries")

    # Extract: group members by key stem so wav + json pair up.
    pending: dict[str, dict] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            if "." not in name:
                continue
            stem, suffix = name.rsplit(".", 1)
            data = tf.extractfile(m).read()
            pending.setdefault(stem, {})[suffix.lower()] = data

    staged_files: list[Path] = []
    rows: list[dict] = []
    n = 0
    for key, parts in pending.items():
        wav_bytes = parts.get("wav") or parts.get("flac")
        if wav_bytes is None or "json" not in parts:
            print(f"  skip {key}: missing wav/json ({list(parts)})")
            continue
        vid = key.replace("/", "_")
        if vid in seen_vids:
            print(f"  skip {key}: duplicate vid across shards")
            continue
        ext = "wav" if "wav" in parts else "flac"
        try:
            recipe = json.loads(parts["json"].decode("utf-8"))
        except Exception as ex:  # noqa: BLE001
            print(f"  skip {key}: bad json ({ex})")
            continue

        captions, windows, qids = parse_moments(recipe)
        out_wav = wav_dir / f"{vid}.{ext}"
        out_wav.write_bytes(wav_bytes)
        dur = wav_duration(wav_bytes)

        seen_vids.add(vid)
        staged_files.append(out_wav)
        rows.append({
            "vid": vid,
            "split": split,
            "shard": shard,
            "gs_path": f"{args.gcs_root.rstrip('/')}/{split}/{out_wav.name}",
            "duration": dur,
            "num_moments": len(windows),
            "captions": captions,
            "moments_sec": windows,
            "qids": qids,
        })
        n += 1

    # Upload this shard's wavs, then record manifest + checkpoint, then prune.
    if not args.no_upload:
        dest = f"{args.gcs_root.rstrip('/')}/{split}"
        print(f"  uploading {len(staged_files)} wavs -> {dest}/ ...", flush=True)
        gsutil_cp(staged_files, dest)

    for row in rows:
        manifest_fh.write(json.dumps(row) + "\n")
    manifest_fh.flush()

    if not args.keep_local and not args.no_upload:
        for f in staged_files:
            f.unlink(missing_ok=True)
        tar_path.unlink(missing_ok=True)

    print(f"[{shard}] staged {n} clips", flush=True)
    return n


def do_inspect(shard: str, args) -> None:
    from collections import Counter

    from huggingface_hub import hf_hub_download

    tar_path = Path(hf_hub_download(
        HF_DATASET, filename=shard, repo_type="dataset",
        local_dir=str(Path(args.scratch) / "tars")))
    suffixes: Counter = Counter()
    first_json = None
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if "." in m.name:
                suffixes[m.name.rsplit(".", 1)[1]] += 1
            if first_json is None and m.name.endswith(".json"):
                first_json = json.loads(tf.extractfile(m).read().decode("utf-8"))
    print(f"shard {shard}: suffixes={dict(suffixes)}")
    print("first json:\n" + json.dumps(first_json, indent=2)[:1500])


def main() -> None:
    args = parse_args()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint = scratch / "staged_shards.txt"
    manifest = scratch / "manifest.jsonl"

    shards = list_shards(args.splits)
    done = load_done(checkpoint)
    todo = [s for s in shards if s not in done]
    if args.shard_limit is not None:
        todo = todo[:args.shard_limit]

    print(f"splits={args.splits} total_shards={len(shards)} done={len(done)} "
          f"todo_this_run={len(todo)} gcs_root={args.gcs_root}")
    if args.dry_run:
        for s in todo:
            print("  would stage:", s)
        return
    if args.inspect:
        do_inspect(todo[0] if todo else shards[0], args)
        return

    # Rebuild seen-vids from the existing manifest so resumes stay deduped.
    seen_vids: set[str] = set()
    if manifest.is_file():
        with manifest.open() as fh:
            for ln in fh:
                try:
                    seen_vids.add(json.loads(ln)["vid"])
                except Exception:  # noqa: BLE001
                    pass

    total = 0
    with manifest.open("a") as manifest_fh, checkpoint.open("a") as ckpt_fh:
        for i, shard in enumerate(todo, 1):
            try:
                total += process_shard(shard, args, seen_vids, manifest_fh)
            except subprocess.CalledProcessError as ex:
                print(f"[{shard}] UPLOAD FAILED ({ex}); stopping. "
                      f"Re-run to resume (check gcloud auth).", file=sys.stderr)
                break
            except Exception as ex:  # noqa: BLE001
                print(f"[{shard}] ERROR: {ex}; skipping shard", file=sys.stderr)
                continue
            ckpt_fh.write(shard + "\n")
            ckpt_fh.flush()
            print(f"  progress: {i}/{len(todo)} shards this run, "
                  f"{total} clips staged", flush=True)

    # Free the hf_hub blob cache dir if we pruned (symlinks -> deleted blobs).
    if not args.keep_local and not args.no_upload:
        cache = scratch / "tars" / ".cache"
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)

    print(f"\nDone. {total} clips staged this run. "
          f"manifest: {manifest}  checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
