#!/usr/bin/env python
"""Load Clotho-Moment (DCASE 2026 Task 6: Audio Moment Retrieval) into FiftyOne.

Task 6 is temporal grounding: given a long audio clip and a text query, return
the ``[start, end]`` moment(s) that match. We turn each clip into a
**spectrogram video** (the original audio muxed under a full-clip spectrogram)
so FiftyOne renders it with the native video timeline + audio playback, the
moments become ``fo.TemporalDetections`` on that timeline, and you can evaluate
retrieval with the native ActivityNet temporal protocol
(``view.evaluate_detections(..., method="activitynet")``).

Clotho-Moment ships as **WebDataset tars** with an undocumented per-sample
schema, so this loader streams via ``datasets`` and *introspects* keys at
runtime. Run with ``--inspect`` first to print the first sample's structure,
confirm the field mapping, then drop ``--inspect`` to build.

Fields written per sample:
    query          StringField              the text query / caption
    ground_truth   fo.TemporalDetections    one "moment" per relevant window
    moments_sec    ListField                raw [[start, end], ...] in seconds

Usage:
    python load_clotho_moment.py --inspect
    python load_clotho_moment.py --max-samples 100 --env-file .env \\
        --cloud-prefix gs://your-bucket/clotho-moment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # -> lib.audio, _lib.env

HF_DATASET = "lighthouse-emnlp2024/Clotho-Moment"

# candidate field names (defensive — schema is not documented)
AUDIO_KEYS = ["wav", "flac", "mp3", "audio"]
QUERY_KEYS = ["query", "caption", "text", "description"]
WINDOW_KEYS = ["relevant_windows", "windows", "timestamps", "moments", "moment"]
DURATION_KEYS = ["duration", "audio_duration"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="clotho-moment")
    p.add_argument("--hf-dataset", default=HF_DATASET)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--inspect", action="store_true",
                   help="print the first streamed sample's structure and exit")
    p.add_argument("--videos-dir", default=str(PROJECT_ROOT / "scratch" / "clotho_moment_videos"))
    p.add_argument("--audio-dir", default=str(PROJECT_ROOT / "scratch" / "clotho_moment_audio"))
    p.add_argument("--scroll", action="store_true",
                   help="use a scrolling spectrum instead of a static one")
    p.add_argument("--cloud-prefix", default=None,
                   help="sample.filepath = <cloud-prefix>/<file>.mp4")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Defensive field extraction
# ---------------------------------------------------------------------------


def _first_present(row: dict, keys):
    """Return (key, value) for the first present/truthy key, searching a nested
    ``json`` dict too."""
    for k in keys:
        if row.get(k) not in (None, "", [], {}):
            return k, row[k]
    nested = row.get("json")
    if isinstance(nested, str):
        try:
            nested = json.loads(nested)
        except Exception:
            nested = None
    if isinstance(nested, dict):
        for k in keys:
            if nested.get(k) not in (None, "", [], {}):
                return k, nested[k]
    return None, None


def _parse_windows(value):
    """Normalize a relevant-windows value into a list of [start, end] floats."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if not isinstance(value, (list, tuple)) or not value:
        return []
    # already [[s,e],...]
    if isinstance(value[0], (list, tuple)):
        return [[float(w[0]), float(w[1])] for w in value if len(w) >= 2]
    # flat [s,e]
    if len(value) >= 2 and isinstance(value[0], (int, float)):
        return [[float(value[0]), float(value[1])]]
    return []


def _write_audio(audio, out_wav: Path) -> Path | None:
    """Materialize a datasets audio value to a .wav on disk. Returns path."""
    import soundfile as sf

    if isinstance(audio, dict):
        if "array" in audio and audio.get("array") is not None:
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_wav, audio["array"], audio["sampling_rate"])
            return out_wav
        if audio.get("path") and Path(audio["path"]).is_file():
            return Path(audio["path"])
        if audio.get("bytes"):
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            out_wav.write_bytes(audio["bytes"])
            return out_wav
    return None


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    from _lib import env
    env.load(args.env_file)

    from datasets import load_dataset

    print(f"Streaming {args.hf_dataset} [{args.hf_split}] ...")
    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)

    if args.inspect:
        row = next(iter(ds))
        print("\n=== first sample keys/types ===")
        for k, v in row.items():
            desc = type(v).__name__
            if isinstance(v, dict):
                desc += f" keys={list(v.keys())}"
            elif isinstance(v, str):
                desc += f" = {v[:120]!r}"
            print(f"  {k}: {desc}")
        ak, _ = _first_present(row, AUDIO_KEYS)
        qk, qv = _first_present(row, QUERY_KEYS)
        wk, wv = _first_present(row, WINDOW_KEYS)
        print("\n=== resolved mapping ===")
        print(f"  audio key:   {ak}")
        print(f"  query key:   {qk} -> {str(qv)[:80]!r}")
        print(f"  window key:  {wk} -> {wv}")
        return

    import fiftyone as fo
    from lib.audio import spectrogram_video, scrolling_spectrogram_video

    render = scrolling_spectrogram_video if args.scroll else spectrogram_video
    audio_dir = Path(args.audio_dir)
    videos_dir = Path(args.videos_dir)

    pending = []  # (sample, windows_sec)
    n = 0
    for row in ds:
        if n >= args.max_samples:
            break
        _, audio = _first_present(row, AUDIO_KEYS)
        if audio is None:
            continue
        key = str(row.get("__key__", f"{n:06d}")).replace("/", "_")
        wav = _write_audio(audio, audio_dir / f"{key}.wav")
        if wav is None:
            continue

        mp4 = videos_dir / f"{key}.mp4"
        if not mp4.is_file():
            render(wav, mp4)

        _, qv = _first_present(row, QUERY_KEYS)
        _, wv = _first_present(row, WINDOW_KEYS)
        windows = _parse_windows(wv)

        filepath = (f"{args.cloud_prefix.rstrip('/')}/{mp4.name}"
                    if args.cloud_prefix else str(mp4))
        sample = fo.Sample(filepath=filepath)
        sample["query"] = str(qv) if qv else ""
        sample["moments_sec"] = windows
        pending.append((sample, windows))
        n += 1
        if n % 25 == 0:
            print(f"  rendered {n} clips")

    print(f"Adding {len(pending)} video samples ...")
    dataset = fo.Dataset(args.dataset_name, overwrite=args.overwrite)
    dataset.add_samples([s for s, _ in pending])
    dataset.persistent = True

    # metadata is needed to convert [start,end] seconds -> support frames
    print("Computing video metadata ...")
    dataset.compute_metadata()

    print("Attaching TemporalDetections ...")
    for sample, windows in zip(dataset, [w for _, w in pending]):
        dets = []
        for s, e in windows:
            try:
                dets.append(
                    fo.TemporalDetection.from_timestamps([s, e], sample=sample,
                                                         label="moment"))
            except Exception as ex:
                print(f"  skip window {[s, e]} on {sample.filepath}: {ex}")
        sample["ground_truth"] = fo.TemporalDetections(detections=dets)
        sample.save()

    dataset.info["source"] = f"Clotho-Moment / DCASE 2026 Task 6 ({args.hf_dataset})"
    dataset.save()

    print("\nDone.")
    print(dataset)
    print("Evaluate retrieval with: "
          "view.evaluate_detections('predictions', gt_field='ground_truth', "
          "method='activitynet', compute_mAP=True)  # results.mAP()")


if __name__ == "__main__":
    main()
