#!/usr/bin/env python
"""Convert a Clotho-Moment split to spectrogram-video with per-query GT.

The `clotho-moment` dataset (built by
`dcase26-task6-amr/build_clotho_moment_dataset.py`) is raw wav
(``media_type="unknown"``) -- no timeline, no `TemporalDetection`s, no native
temporal evaluation. This renders each clip as a spectrogram video (original
audio muxed in, via `lib.audio.spectrogram_video`) so FiftyOne treats it as
``media_type="video"`` and unlocks the native timeline + ActivityNet-style
temporal evaluation.

Reads directly from `dcase26-task6-amr/scratch/manifest.jsonl` (authoritative
per-clip metadata: vid/split/duration/num_moments/captions/moments_sec/qids)
rather than re-querying the wav dataset.

A clip can have 0-3 independent (caption, window, qid) triples -- parallel
lists, one entry per relevant moment for a DIFFERENT text query. Each becomes
its own `fo.TemporalDetection` labeled with `str(qid)` (not a generic
"moment") so that a later `evaluate_detections(..., classwise=True)` only
matches a prediction to the ground truth for the SAME query, not any
overlapping window on the same clip.

Usage:
    python loaders/convert_clotho_moment_to_video.py --env-file .env \\
        --split valid --wav-dir scratch/clotho_moment_wav/valid \\
        --videos-dir scratch/clotho_moment_video/valid \\
        --cloud-prefix gs://your-bucket/clotho-moment-video/valid
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # -> lib.audio, _lib.env

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "dcase26-task6-amr" / "scratch" / "manifest.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="clotho-moment-video")
    p.add_argument("--split", default="valid")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--wav-dir", required=True,
                   help="local dir of <vid>.wav files for this split")
    p.add_argument("--videos-dir", required=True,
                   help="local dir to render <vid>.mp4 files into")
    p.add_argument("--cloud-prefix", required=True,
                   help="sample.filepath = <cloud-prefix>/<vid>.mp4")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel ffmpeg renders (spectrogram_video is CPU-bound)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-render", action="store_true",
                   help="videos already rendered; only (re)build the dataset")
    p.add_argument("--skip-upload", action="store_true",
                   help="mp4s already uploaded to --cloud-prefix; skip re-uploading")
    return p.parse_args()


def read_manifest_rows(manifest: Path, split: str, max_samples):
    rows = []
    with manifest.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split:
                continue
            rows.append(row)
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _render_one(args_tuple):
    """Render a single clip. Run in a worker process -- import here."""
    wav_path, mp4_path = args_tuple
    if mp4_path.is_file():
        return mp4_path.name, None
    from lib.audio import spectrogram_video

    try:
        spectrogram_video(wav_path, mp4_path)
        return mp4_path.name, None
    except Exception as e:
        return mp4_path.name, str(e)


def render_videos(rows, wav_dir: Path, videos_dir: Path, workers: int):
    videos_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(wav_dir / f"{r['vid']}.wav", videos_dir / f"{r['vid']}.mp4") for r in rows]
    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_render_one, job): job for job in jobs}
        for fut in as_completed(futures):
            name, err = fut.result()
            done += 1
            if err:
                failures.append((name, err))
                print(f"  FAILED {name}: {err}")
            if done % 200 == 0 or done == len(jobs):
                print(f"  rendered {done}/{len(jobs)} ({len(failures)} failed)")
    return failures


def upload_videos(videos_dir: Path, cloud_prefix: str) -> None:
    """Bulk-upload rendered mp4s to GCS so the built dataset's `gs://`
    filepaths resolve. Metadata is computed from the local mp4 (see
    `build_dataset`), not read back from the cloud copy, so this upload's
    timing relative to dataset build no longer matters.
    """
    import subprocess

    subprocess.run(
        ["bash", "-c", f"gcloud storage cp -r {videos_dir}/* {cloud_prefix}/"],
        check=True,
    )


def build_dataset(rows, args):
    import fiftyone as fo

    dataset = fo.Dataset(args.dataset_name, overwrite=args.overwrite)
    videos_dir = Path(args.videos_dir)
    samples = []
    for r in rows:
        filepath = f"{args.cloud_prefix.rstrip('/')}/{r['vid']}.mp4"
        sample = fo.Sample(filepath=filepath)
        sample["split"] = r["split"]
        sample["vid"] = r["vid"]
        sample["duration"] = r["duration"]
        sample["num_moments"] = r["num_moments"]
        sample["captions"] = r["captions"]
        sample["moments_sec"] = r["moments_sec"]
        sample["qids"] = r["qids"]
        # Compute metadata from the local render, not the cloud copy -- avoids
        # a GCS-propagation race right after upload (see `upload_videos`).
        sample.metadata = fo.VideoMetadata.build_for(str(videos_dir / f"{r['vid']}.mp4"))
        samples.append(sample)

    print(f"Adding {len(samples)} video samples ...")
    dataset.add_samples(samples)
    dataset.persistent = True

    print("Attaching per-query ground_truth TemporalDetections ...")
    n_dets = 0
    for sample in dataset.iter_samples(autosave=True, progress=True):
        dets = []
        for caption, window, qid in zip(
            sample["captions"], sample["moments_sec"], sample["qids"]
        ):
            s, e = window
            try:
                det = fo.TemporalDetection.from_timestamps(
                    [s, e], sample=sample, label=str(qid)
                )
                det["query"] = caption
                dets.append(det)
            except Exception as ex:
                print(f"  skip window {[s, e]} on {sample.vid} (qid={qid}): {ex}")
        sample["ground_truth"] = fo.TemporalDetections(detections=dets)
        n_dets += len(dets)

    dataset.info["source"] = "Clotho-Moment / DCASE 2026 Task 6 (spectrogram-video)"
    dataset.save()
    print(f"\nDone. {len(dataset)} samples, {n_dets} ground-truth detections.")
    print(dataset)


def main() -> None:
    args = parse_args()
    from _lib import env
    env.load(args.env_file)

    manifest = Path(args.manifest)
    rows = read_manifest_rows(manifest, args.split, args.max_samples)
    print(f"{len(rows)} clips for split={args.split!r} from {manifest}")

    wav_dir = Path(args.wav_dir)
    videos_dir = Path(args.videos_dir)

    if not args.skip_render:
        missing_wavs = [r["vid"] for r in rows if not (wav_dir / f"{r['vid']}.wav").is_file()]
        if missing_wavs:
            raise FileNotFoundError(
                f"{len(missing_wavs)} wavs missing under {wav_dir} "
                f"(e.g. {missing_wavs[0]}.wav). Download the split's wavs first."
            )
        print(f"Rendering {len(rows)} spectrogram videos with {args.workers} workers ...")
        failures = render_videos(rows, wav_dir, videos_dir, args.workers)
        if failures:
            print(f"\n{len(failures)} clips failed to render; excluding from dataset.")
            failed_vids = {name.rsplit(".", 1)[0] for name, _ in failures}
            rows = [r for r in rows if r["vid"] not in failed_vids]

    if not args.skip_upload:
        print(f"Uploading {len(rows)} mp4s to {args.cloud_prefix}/ ...")
        upload_videos(videos_dir, args.cloud_prefix)

    build_dataset(rows, args)


if __name__ == "__main__":
    main()
