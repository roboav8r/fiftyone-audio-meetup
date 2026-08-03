#!/usr/bin/env python
"""Compute sliding-window CLAP/MS-CLAP embeddings for Clotho-Moment clips.

CLAP/MS-CLAP have a ~10 s native window, so a 60 s clip is embedded as a
sequence of overlapping windows (default 10 s window, 1 s hop) → an
``(N, dim)`` matrix (dim=512 for CLAP, 1024 for MS-CLAP) that is the shared
substrate for:
  - the windows dataset + embeddings-search panel (retrieval),
  - the threshold retrieval algorithms (A/B/hybrid),
  - the MCAP CLAP-similarity / eventness Plot streams.

Per clip we persist two sidecars and upload them to GCS, namespaced by
backend so CLAP and MS-CLAP sidecars never collide:
  <vid>.npy   float32 [N, dim]  L2-normalized window embeddings
  <vid>.json  {vid, split, backend, sr, win_s, hop_s, starts[N], ends[N]}

Reads wavs from the already-staged <wav-gcs>/<split>/ (see
`stage_clotho_moment.py`). Runs locally on the GPU (audio-emb env -- has
both `transformers` for CLAP and the `msclap` package for MS-CLAP). No
FiftyOne import.

Usage:
    python clap_windows.py --wav-gcs gs://your-bucket/clotho-moment \\
        --split valid --backend clap --limit 50
    python clap_windows.py --wav-gcs gs://your-bucket/clotho-moment \\
        --split valid --backend msclap --vids Singapore_00_600
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root → clap/

BACKENDS = {}  # backend name -> (CLAP_SAMPLE_RATE, embed_audio_arrays)


def _load_backend(name: str):
    if name in BACKENDS:
        return BACKENDS[name]
    if name == "clap":
        from clap.clap import CLAP_SAMPLE_RATE, embed_audio_arrays
    elif name == "msclap":
        from clap.msclap import embed_audio_arrays

        CLAP_SAMPLE_RATE = 48000  # msclap resamples internally; feed it 48kHz
    else:
        raise ValueError(f"unknown backend {name!r} (choices: clap, msclap)")
    BACKENDS[name] = (CLAP_SAMPLE_RATE, embed_audio_arrays)
    return BACKENDS[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="valid")
    p.add_argument("--backend", default="clap", choices=["clap", "msclap"])
    p.add_argument("--manifest", default=str(Path(__file__).resolve().parents[1] / "scratch" / "manifest.jsonl"))
    p.add_argument("--vids", nargs="+", default=None, help="explicit vids (else from manifest)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-negatives", action="store_true",
                   help="skip clips with num_moments=0 (nothing to query against)")
    p.add_argument("--win-s", type=float, default=10.0)
    p.add_argument("--hop-s", type=float, default=1.0)
    p.add_argument("--wav-gcs", required=True,
                   help="GCS prefix wavs were staged to, e.g. gs://your-bucket/clotho-moment "
                        "(see stage_clotho_moment.py --gcs-root)")
    p.add_argument("--local-wav-dir", default=None,
                   help="dir of already-downloaded <vid>.wav (skips per-clip GCS download)")
    p.add_argument("--out-gcs", default=None,
                   help="GCS prefix to upload sidecars to (defaults to <wav-gcs>-clap-windows/<backend>)")
    p.add_argument("--scratch", default=str(Path(__file__).resolve().parents[1] / "scratch" / "clap"))
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes -- the HF/msclap feature "
                        "extractors are CPU-bound per-sample, so this scales "
                        "close to linearly with CPU cores")
    return p.parse_args()


def slide_windows(y: np.ndarray, sr: int, win_s: float, hop_s: float):
    """Yield (start_s, end_s, window_array). Last partial window is kept if it
    covers >= half a hop of new audio; short clips yield a single full-length window."""
    win = int(round(win_s * sr))
    hop = int(round(hop_s * sr))
    n = len(y)
    if n <= win:
        return [(0.0, n / sr, y)]
    out = []
    start = 0
    while start < n:
        end = min(start + win, n)
        out.append((start / sr, end / sr, y[start:end]))
        if end >= n:
            break
        start += hop
    return out


def vids_from_manifest(manifest: Path, split: str, limit, explicit, skip_negatives: bool):
    if explicit:
        return list(explicit)
    vids = []
    with manifest.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            if row.get("split") != split:
                continue
            if skip_negatives and row.get("num_moments", 0) == 0:
                continue
            vids.append(row["vid"])
            if limit and len(vids) >= limit:
                break
    return vids


def embed_clip(wav_path: Path, win_s: float, hop_s: float, batch_size: int,
                sample_rate: int, embed_fn):
    import soundfile as sf

    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != sample_rate:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    wins = slide_windows(y, sr, win_s, hop_s)
    starts = [w[0] for w in wins]
    ends = [w[1] for w in wins]
    embs = embed_fn([w[2] for w in wins], batch_size=batch_size)
    return np.asarray(starts, np.float32), np.asarray(ends, np.float32), embs


def _process_one(job):
    """Runs in a worker process. Imports/loads the backend model lazily
    (per-process) so CUDA context init happens after fork, not before."""
    (vid, split, backend, wav_gcs, local_wav_dir, scratch_str,
     win_s, hop_s, batch_size) = job
    scratch = Path(scratch_str)
    npy = scratch / f"{vid}.npy"
    meta = scratch / f"{vid}.json"
    if npy.is_file() and meta.is_file():
        try:
            return vid, np.load(npy).shape, None
        except Exception:
            pass  # corrupt sidecar; fall through and recompute

    sample_rate, embed_fn = _load_backend(backend)

    if local_wav_dir:
        local_wav = Path(local_wav_dir) / f"{vid}.wav"
        if not local_wav.is_file():
            return vid, None, f"missing local wav {local_wav}"
    else:
        local_wav = scratch / f"{vid}.wav"
        if not local_wav.is_file():
            gs_wav = f"{wav_gcs}/{split}/{vid}.wav"
            subprocess.run(["gsutil", "-q", "cp", gs_wav, str(local_wav)], check=True)

    try:
        starts, ends, embs = embed_clip(local_wav, win_s, hop_s, batch_size, sample_rate, embed_fn)
    except Exception as e:
        return vid, None, str(e)

    np.save(npy, embs)
    meta.write_text(json.dumps({
        "vid": vid, "split": split, "backend": backend, "sr": sample_rate,
        "win_s": win_s, "hop_s": hop_s,
        "starts": starts.tolist(), "ends": ends.tolist(),
        "n_windows": int(embs.shape[0]), "dim": int(embs.shape[1]),
    }))
    if not local_wav_dir:
        local_wav.unlink(missing_ok=True)  # only clean up what we downloaded ourselves
    return vid, embs.shape, None


def main() -> None:
    args = parse_args()
    out_gcs = args.out_gcs or f"{args.wav_gcs}-clap-windows/{args.backend}"

    scratch = Path(args.scratch) / args.backend / args.split
    scratch.mkdir(parents=True, exist_ok=True)
    vids = vids_from_manifest(Path(args.manifest), args.split, args.limit, args.vids,
                               args.skip_negatives)
    print(f"backend={args.backend}  clip count: {len(vids)}  "
          f"win={args.win_s}s hop={args.hop_s}s  workers={args.workers}")

    jobs = [
        (vid, args.split, args.backend, args.wav_gcs, args.local_wav_dir, str(scratch),
         args.win_s, args.hop_s, args.batch_size)
        for vid in vids
    ]

    failures = []
    done = 0
    if args.workers <= 1:
        results = (_process_one(job) for job in jobs)
        for vid, shape, err in results:
            done += 1
            if err:
                failures.append((vid, err))
                print(f"  FAILED {vid}: {err}")
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {vid}: {shape}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process_one, job): job for job in jobs}
            for fut in as_completed(futures):
                vid, shape, err = fut.result()
                done += 1
                if err:
                    failures.append((vid, err))
                    print(f"  FAILED {vid}: {err}")
                if done % 50 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}  {vid}: {shape}")

    if not args.no_upload:
        print(f"Uploading {len(vids) - len(failures)} sidecar pairs to {out_gcs}/{args.split}/ ...")
        subprocess.run(
            ["bash", "-c", f"gcloud storage cp {scratch}/*.npy {scratch}/*.json "
                            f"{out_gcs}/{args.split}/"],
            check=True,
        )

    if failures:
        print(f"\n{len(failures)} clips FAILED: {[v for v, _ in failures][:10]}"
              f"{' ...' if len(failures) > 10 else ''}")
    print("done.")


if __name__ == "__main__":
    main()
