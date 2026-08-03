#!/usr/bin/env python
"""Load the Clotho audio-captioning dataset into FiftyOne.

Clotho (Drossos et al.) pairs 15–30s audio clips with **five** human captions
each. This loader produces the schema the reference captioning operators
expect — a `.wav` sample (media_type ``unknown``) with a `ground_truth`
``fo.Classification`` caption — so you can run, end to end:

    @ehofesmann/apply_audio_captioning_model   (CoNeTTE)   -> predictions
    @ehofesmann/evaluate_audio_captions        (aac_metrics) -> SPIDEr/FENSE/...

Fields written per sample:
    ground_truth   fo.Classification   primary reference caption (caption_1)
    captions       ListField(String)   all 5 reference captions
    split          StringField          development | validation | evaluation
    spectrogram    StringField          precomputed mel-spectrogram PNG (opt.)

Three ways to get the data (in priority order):
    1. --captions-csv + --audio-dir   (a Clotho checkout you already have)
    2. --hf-dataset <repo> [--hf-split <split>]   (stream via `datasets`)
    3. (default) auto-download the chosen --split from Zenodo (needs py7zr)

Usage:
    python load_clotho.py --split validation --max-samples 200
    python load_clotho.py --env-file .env \\
        --captions-csv /data/clotho/clotho_captions_validation.csv \\
        --audio-dir /data/clotho/validation \\
        --cloud-prefix gs://your-bucket/clotho/validation
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # -> lib.audio, _lib.env

# Clotho v2.1 on Zenodo (record 4783391). Audio archives are .7z.
ZENODO_RECORD = "4783391"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
CLOTHO_FILES = {
    "development": ("clotho_audio_development.7z", "clotho_captions_development.csv"),
    "validation": ("clotho_audio_validation.7z", "clotho_captions_validation.csv"),
    "evaluation": ("clotho_audio_evaluation.7z", "clotho_captions_evaluation.csv"),
}
CAPTION_COLS = [f"caption_{i}" for i in range(1, 6)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="clotho")
    p.add_argument("--split", default="validation",
                   choices=list(CLOTHO_FILES.keys()))
    p.add_argument("--captions-csv", default=None,
                   help="path to a clotho_captions_<split>.csv (skips download)")
    p.add_argument("--audio-dir", default=None,
                   help="dir of .wav files matching the CSV file_name column")
    p.add_argument("--hf-dataset", default=None,
                   help="HuggingFace dataset repo id to stream instead of Zenodo")
    p.add_argument("--hf-split", default="test")
    p.add_argument("--download-dir", default=str(PROJECT_ROOT / "scratch" / "clotho"))
    p.add_argument("--spectrograms-dir",
                   default=str(PROJECT_ROOT / "scratch" / "clotho_spectrograms"))
    p.add_argument("--no-spectrograms", action="store_true")
    p.add_argument("--cloud-prefix", default=None,
                   help="sample.filepath = <cloud-prefix>/<file_name>")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def acquire_from_zenodo(split: str, download_dir: Path) -> tuple[Path, Path]:
    """Download + extract the chosen Clotho split. Returns (captions_csv, audio_dir)."""
    audio_7z, captions_csv = CLOTHO_FILES[split]
    csv_path = download_dir / captions_csv
    archive = download_dir / audio_7z
    _download(f"{ZENODO_BASE}/{captions_csv}?download=1", csv_path)
    _download(f"{ZENODO_BASE}/{audio_7z}?download=1", archive)

    audio_dir = download_dir / split
    if not audio_dir.is_dir():
        try:
            import py7zr
        except ImportError as e:
            raise RuntimeError(
                "py7zr is required to extract Clotho .7z archives "
                "(`pip install py7zr`), or pass --audio-dir to an existing "
                "extract."
            ) from e
        print(f"Extracting {archive} ...")
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(download_dir)
        # Clotho archives extract to a folder named after the split.
        if not audio_dir.is_dir():
            # fall back to whatever single dir was produced
            cands = [d for d in download_dir.iterdir() if d.is_dir()]
            if len(cands) == 1:
                audio_dir = cands[0]
    return csv_path, audio_dir


def read_caption_rows(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            captions = [row[c].strip() for c in CAPTION_COLS if row.get(c)]
            yield row["file_name"], captions


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_from_files(args, fo, mel_spectrogram_png):
    """Path 1/3: build from a captions CSV + local audio dir."""
    if args.captions_csv and args.audio_dir:
        csv_path, audio_dir = Path(args.captions_csv), Path(args.audio_dir)
    else:
        csv_path, audio_dir = acquire_from_zenodo(args.split, Path(args.download_dir))

    rows = list(read_caption_rows(csv_path))
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"Building {len(rows)} samples from {csv_path}")

    spec_dir = Path(args.spectrograms_dir)
    samples = []
    for file_name, captions in rows:
        local_wav = audio_dir / file_name
        if args.cloud_prefix:
            filepath = f"{args.cloud_prefix.rstrip('/')}/{file_name}"
        else:
            filepath = str(local_wav)

        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Classification(label=captions[0] if captions else "")
        sample["captions"] = captions
        sample["split"] = args.split

        if not args.no_spectrograms and local_wav.is_file():
            png = spec_dir / (Path(file_name).stem + ".png")
            if not png.is_file():
                mel_spectrogram_png(local_wav, png)
            sample["spectrogram"] = str(png)

        samples.append(sample)
    return samples


def build_from_hf(args, fo, mel_spectrogram_png):
    """Path 2/3: stream from a HuggingFace dataset, defensively mapping fields."""
    import soundfile as sf
    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    audio_out = Path(args.download_dir) / "hf_audio"
    audio_out.mkdir(parents=True, exist_ok=True)
    spec_dir = Path(args.spectrograms_dir)

    caption_keys = ["caption", "captions", "caption_1", "text", "ground_truth"]
    samples = []
    for i, row in enumerate(ds):
        if args.max_samples and i >= args.max_samples:
            break
        # audio
        audio = row.get("audio") or row.get("wav") or row.get("flac")
        if isinstance(audio, dict) and "array" in audio:
            wav = audio_out / f"{i:06d}.wav"
            sf.write(wav, audio["array"], audio["sampling_rate"])
        elif isinstance(audio, dict) and audio.get("path"):
            wav = Path(audio["path"])
        else:
            continue
        # captions
        caps = None
        for k in caption_keys:
            if row.get(k):
                caps = row[k]
                break
        if isinstance(caps, str):
            caps = [caps]
        caps = caps or [""]

        filepath = (f"{args.cloud_prefix.rstrip('/')}/{wav.name}"
                    if args.cloud_prefix else str(wav))
        sample = fo.Sample(filepath=filepath)
        sample["ground_truth"] = fo.Classification(label=caps[0])
        sample["captions"] = list(caps)
        if not args.no_spectrograms and wav.is_file():
            png = spec_dir / (wav.stem + ".png")
            if not png.is_file():
                mel_spectrogram_png(wav, png)
            sample["spectrogram"] = str(png)
        samples.append(sample)
    print(f"Built {len(samples)} samples from HF dataset {args.hf_dataset}")
    return samples


def main() -> None:
    args = parse_args()
    from _lib import env
    env.load(args.env_file)

    import fiftyone as fo
    from lib.audio import mel_spectrogram_png

    if args.hf_dataset:
        samples = build_from_hf(args, fo, mel_spectrogram_png)
    else:
        samples = build_from_files(args, fo, mel_spectrogram_png)

    dataset = fo.Dataset(args.dataset_name, overwrite=args.overwrite)
    dataset.add_samples(samples)
    dataset.persistent = True
    dataset.info["source"] = "Clotho v2.1 (Zenodo 4783391)"
    dataset.save()

    print("\nDone.")
    print(dataset)
    print("Next: run @ehofesmann/apply_audio_captioning_model (CoNeTTE) then "
          "@ehofesmann/evaluate_audio_captions.")


if __name__ == "__main__":
    main()
