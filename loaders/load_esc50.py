#!/usr/bin/env python
"""Load the ESC-50 environmental-sound dataset into FiftyOne.

ESC-50 is 2,000 five-second clips across 50 classes (5 balanced folds).
FiftyOne has no native audio media type, so each clip becomes a sample whose
``filepath`` is the ``.wav`` (media_type ``unknown``); the custom spectrogram
sample renderer draws it in the grid. We also precompute a mel-spectrogram PNG
per clip (``spectrogram`` field) as a renderer fallback / tabular thumbnail.

Fields written per sample:
    category        fo.Classification   fine label (50 classes)
    major_category  fo.Classification   coarse label (5 groups)
    fold            IntField            1..5 (ESC-50 CV fold)
    target          IntField            0..49 (class id)
    esc10           BooleanField        member of the ESC-10 subset
    split           StringField         "test" if fold == 5 else "train"
    spectrogram     StringField         path to precomputed PNG (optional)

Usage:
    # local (OSS or FOE, whatever's configured in .env / your shell)
    python load_esc50.py --max-samples 200

    # against a hosted deployment, with media served from a cloud bucket
    python load_esc50.py --env-file .env \\
        --cloud-prefix gs://your-bucket/esc50

Source media must be reachable by the deployment your dataset lives on, so
upload ESC-50's audio/ dir to a bucket the deployment can read and pass
--cloud-prefix accordingly.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# --- repo / project imports (before `import fiftyone`) ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # -> lib.audio, _lib.env

ESC50_ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

# ESC-50's 50 fine classes grouped into the 5 documented major categories.
MAJOR_CATEGORY = {
    # Animals
    **{c: "Animals" for c in [
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects",
        "sheep", "crow"]},
    # Natural soundscapes & water sounds
    **{c: "Natural soundscapes & water sounds" for c in [
        "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
        "water_drops", "wind", "pouring_water", "toilet_flush",
        "thunderstorm"]},
    # Human, non-speech sounds
    **{c: "Human, non-speech sounds" for c in [
        "crying_baby", "sneezing", "clapping", "breathing", "coughing",
        "footsteps", "laughing", "brushing_teeth", "snoring",
        "drinking_sipping"]},
    # Interior/domestic sounds
    **{c: "Interior/domestic sounds" for c in [
        "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
        "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
        "clock_tick", "glass_breaking"]},
    # Exterior/urban noises
    **{c: "Exterior/urban noises" for c in [
        "helicopter", "chainsaw", "siren", "car_horn", "engine", "train",
        "church_bells", "airplane", "fireworks", "hand_saw"]},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="ESC-50")
    p.add_argument("--source-dir", default=None,
                   help="path to an existing ESC-50 checkout (skips download)")
    p.add_argument("--download-dir", default=str(PROJECT_ROOT / "scratch" / "esc50"),
                   help="where to download/extract ESC-50")
    p.add_argument("--spectrograms-dir",
                   default=str(PROJECT_ROOT / "scratch" / "esc50_spectrograms"))
    p.add_argument("--no-spectrograms", action="store_true",
                   help="skip precomputing spectrogram PNGs")
    p.add_argument("--cloud-prefix", default=None,
                   help="if set, sample.filepath = <cloud-prefix>/<file> "
                        "(media must already be uploaded there, flat layout)")
    p.add_argument("--esc10-only", action="store_true",
                   help="load only the 10-class ESC-10 subset")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="delete any existing dataset of this name first")
    return p.parse_args()


def ensure_esc50(source_dir, download_dir) -> Path:
    """Return the ESC-50 root dir (with audio/ and meta/), downloading if needed."""
    if source_dir:
        root = Path(source_dir)
        if not (root / "meta" / "esc50.csv").is_file():
            raise FileNotFoundError(f"{root} is not an ESC-50 checkout (no meta/esc50.csv)")
        return root

    download_dir = Path(download_dir)
    root = download_dir / "ESC-50-master"
    if (root / "meta" / "esc50.csv").is_file():
        print(f"Using cached ESC-50 at {root}")
        return root

    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "ESC-50-master.zip"
    if not zip_path.is_file():
        print(f"Downloading ESC-50 (~600 MB) from {ESC50_ZIP_URL} ...")
        urllib.request.urlretrieve(ESC50_ZIP_URL, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(download_dir)
    if not (root / "meta" / "esc50.csv").is_file():
        raise RuntimeError(f"Extraction did not yield {root}/meta/esc50.csv")
    return root


def read_meta(root: Path):
    """Yield dict rows from meta/esc50.csv."""
    import csv

    with open(root / "meta" / "esc50.csv", newline="") as f:
        yield from csv.DictReader(f)


def main() -> None:
    args = parse_args()

    from _lib import env
    env.load(args.env_file)  # override=True by default; before fiftyone

    import fiftyone as fo
    from lib.audio import mel_spectrogram_png

    root = ensure_esc50(args.source_dir, args.download_dir)
    audio_dir = root / "audio"

    rows = list(read_meta(root))
    if args.esc10_only:
        rows = [r for r in rows if r["esc10"].lower() == "true"]
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"Building {len(rows)} samples for dataset '{args.dataset_name}'")

    spec_dir = Path(args.spectrograms_dir)
    make_spec = not args.no_spectrograms

    samples = []
    for i, row in enumerate(rows):
        filename = row["filename"]
        category = row["category"]
        local_wav = audio_dir / filename

        if args.cloud_prefix:
            # cloud bucket is flat (wavs alongside the .mp4s), no audio/ subdir
            filepath = f"{args.cloud_prefix.rstrip('/')}/{filename}"
        else:
            filepath = str(local_wav)

        sample = fo.Sample(filepath=filepath)
        sample["category"] = fo.Classification(label=category)
        sample["major_category"] = fo.Classification(
            label=MAJOR_CATEGORY.get(category, "unknown"))
        sample["target"] = int(row["target"])
        sample["fold"] = int(row["fold"])
        sample["esc10"] = row["esc10"].lower() == "true"
        sample["split"] = "test" if int(row["fold"]) == 5 else "train"
        sample.tags.append(f"fold{row['fold']}")

        if make_spec:
            png = spec_dir / (Path(filename).stem + ".png")
            if not png.is_file():
                mel_spectrogram_png(local_wav, png)
            sample["spectrogram"] = str(png)

        samples.append(sample)
        if (i + 1) % 200 == 0:
            print(f"  prepared {i + 1}/{len(rows)}")

    dataset = fo.Dataset(args.dataset_name, overwrite=args.overwrite)
    dataset.add_samples(samples)
    dataset.persistent = True

    # Friendly defaults for browsing audio: surface the coarse + fine labels.
    dataset.info["source"] = "ESC-50 (github.com/karolpiczak/ESC-50)"
    dataset.save()

    print("\nDone.")
    print(dataset)
    print("Class distribution (major_category):")
    print(dataset.count_values("major_category.label"))


if __name__ == "__main__":
    main()
