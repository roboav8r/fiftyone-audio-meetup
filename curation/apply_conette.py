#!/usr/bin/env python
"""Run CoNeTTE audio captioning over a Clotho dataset and write predictions.

Why a script (vs. the reference ``apply_audio_captioning_model`` operator):
the operator calls ``sample_collection.values("filepath")`` and hands those
strings straight to the CoNeTTE model. If ``filepath`` is a cloud URI (e.g.
``gs://...``), CoNeTTE (like CLAP in
``fiftyone-audio-toolkit/scripts/embed_clap.py``) can't read that directly,
and the delegated runner isn't guaranteed to have ``conette``/``torch==1.13.1``
installed. So, same pattern as that script: read audio from a LOCAL copy
(keyed by filepath basename) on this GPU box, and write results back to the
dataset via ``set_values``.

The CoNeTTE dependency pins ``torch==1.13.1`` / ``transformers==4.30.2``
exactly, which is why this runs in its own env (``fo-clotho-caption``) rather
than whatever env already has CLAP/MS-CLAP's newer torch installed.

Writes the SAME field names the reference operator uses, so results stay
compatible with it (and with ``evaluate_captions.py``, which expects
"predictions"/"ground_truth"):
    <label-field> (default "predictions")        fo.Classification(label, confidence)
    multi_<label-field>                            fo.Classifications (mult_cands/mult_lprobs)
    tags_<label-field>                             ListField(String) raw CoNeTTE tags

Usage (in the fo-clotho-caption env, with the FiftyOne SDK installed):
    python curation/apply_conette.py --env-file .env \\
        --dataset-name clotho --audio-dir scratch/clotho/validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # -> _lib.env

MODEL_ID = "Labbeti/conette"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="clotho")
    p.add_argument("--audio-dir", required=True,
                   help="local dir of .wav files (matched by filepath basename)")
    p.add_argument("--label-field", default="predictions")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing label field instead of erroring")
    return p.parse_args()


def load_conette(device: str):
    from conette import CoNeTTEConfig, CoNeTTEModel

    config = CoNeTTEConfig.from_pretrained(MODEL_ID)
    model = CoNeTTEModel.from_pretrained(MODEL_ID, config=config)
    return model.to(device).eval()


def caption_batch(model, paths: list[str]):
    """Run CoNeTTE on a batch of local wav paths.

    Returns three dicts keyed by ``path``: primary prediction, all beam
    candidates, and raw tags -- mirroring the reference operator's
    ``apply_conette`` batch handler.
    """
    import fiftyone as fo

    outputs = model(paths)
    pred, preds, tags = {}, {}, {}
    for i, path in enumerate(paths):
        pred[path] = fo.Classification(
            label=outputs["cands"][i], confidence=float(outputs["lprobs"][i])
        )
        preds[path] = fo.Classifications(
            classifications=[
                fo.Classification(label=c, confidence=float(p))
                for c, p in zip(outputs["mult_cands"][i], outputs["mult_lprobs"][i])
            ]
        )
        tags[path] = list(outputs["tags"][i])
    return pred, preds, tags


def main() -> None:
    args = parse_args()
    from _lib import env
    env.load(args.env_file)

    import fiftyone as fo
    import fiftyone.core.utils as fou

    dataset = fo.load_dataset(args.dataset_name)
    if args.label_field in dataset.get_field_schema() and not args.overwrite:
        raise ValueError(
            f"Field '{args.label_field}' already exists. Pass --overwrite to "
            "replace it."
        )

    view = dataset.limit(args.max_samples) if args.max_samples else dataset
    ids, filepaths = view.values(["id", "filepath"])

    audio_dir = Path(args.audio_dir)
    local_paths = [str(audio_dir / Path(fp).name) for fp in filepaths]
    missing = [p for p in local_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} local wavs not found under {audio_dir} "
            f"(e.g. {missing[0]}). Point --audio-dir at the extracted Clotho audio."
        )

    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CoNeTTE ({MODEL_ID}) on {device} ...")
    model = load_conette(device)

    pred_by_path, preds_by_path, tags_by_path = {}, {}, {}
    print(f"Captioning {len(ids)} clips (batch={args.batch_size}) ...")
    n_done = 0
    for batch_ids, batch_paths in zip(
        fou.iter_batches(ids, args.batch_size),
        fou.iter_batches(local_paths, args.batch_size),
    ):
        try:
            pred, preds, tags = caption_batch(model, list(batch_paths))
        except Exception as e:
            print(f"  batch failed ({batch_paths[0]}...): {e}")
            continue
        for sid, path in zip(batch_ids, batch_paths):
            pred_by_path[sid] = pred[path]
            preds_by_path[sid] = preds[path]
            tags_by_path[sid] = tags[path]
        n_done += len(batch_ids)
        if n_done % (args.batch_size * 10) == 0 or n_done == len(ids):
            print(f"  {n_done}/{len(ids)}")

    dataset.set_values(args.label_field, pred_by_path, key_field="id")
    dataset.set_values(f"multi_{args.label_field}", preds_by_path, key_field="id")
    dataset.set_values(f"tags_{args.label_field}", tags_by_path, key_field="id")

    n_missing = len(ids) - len(pred_by_path)
    print(f"\nDone. Wrote '{args.label_field}' for {len(pred_by_path)}/{len(ids)} "
          f"samples ({n_missing} failed batches skipped).")
    print("Next: python curation/evaluate_captions.py "
          f"--dataset-name {args.dataset_name} --pred-field {args.label_field}")


if __name__ == "__main__":
    main()
