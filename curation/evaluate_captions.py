#!/usr/bin/env python
"""Evaluate generated captions against ground truth with aac_metrics.

Mirrors the reference ``evaluate_audio_captions`` operator's ``execute()``
body field-for-field, so results stay compatible with it, but writes via
``set_values(..., key_field="id")`` instead of the operator's positional
``view.set_values(...)`` -- alignment-safe regardless of view ordering.

Runs in its own env (``fo-clotho-eval``) because aac_metrics pulls in
sentence-transformers/msclap, which want a much newer torch/transformers
than the pinned ``torch==1.13.1`` CoNeTTE needs in ``apply_conette.py``'s env
-- see that script's docstring.

First run downloads aac_metrics' external resources (spaCy model, METEOR
jar + Java, FENSE's sentence-transformer checkpoint): run
``aac-metrics-download`` first if this errors on missing resources.

Writes (top-level AND under ``predictions.*``, matching the reference op):
    spider, spider_fl, vocab, fense, sbert_sim, cider_d, spice

Usage (in the fo-clotho-eval env, with the FiftyOne SDK installed):
    python curation/evaluate_captions.py --env-file .env \\
        --dataset-name clotho
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # -> _lib.env


def _patch_java_version_check() -> None:
    """conda-forge's openjdk reports versions like "11.0.30-internal", which
    isn't valid PEP 440 and crashes aac_metrics' packaging.version.Version()
    parse. Strip the vendor suffix before it hits that parser -- the Java
    version itself (11, within aac_metrics' supported 8-13 range) is fine.
    """
    from aac_metrics.utils import checks

    orig = checks._check_java_version

    def patched(version_str: str, min_major: int, max_major: int) -> bool:
        version_str = version_str.split("-")[0].split("+")[0]
        return orig(version_str, min_major, max_major)

    checks._check_java_version = patched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None,
                   help="env file to load before connecting (defaults to this repo's .env)")
    p.add_argument("--dataset-name", default="clotho")
    p.add_argument("--pred-field", default="predictions")
    p.add_argument("--gt-field", default="ground_truth")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from _lib import env
    env.load(args.env_file)

    import fiftyone as fo
    _patch_java_version_check()
    from aac_metrics import Evaluate

    dataset = fo.load_dataset(args.dataset_name)
    view = dataset.exists(args.pred_field).exists(args.gt_field)
    print(f"Evaluating {len(view)}/{len(dataset)} samples "
          f"(missing '{args.pred_field}' or '{args.gt_field}' are skipped)")

    ids, candidates, gt_labels = view.values(["id", f"{args.pred_field}.label",
                                               f"{args.gt_field}.label"])
    mult_references = [[l] for l in gt_labels]

    evaluate = Evaluate(metrics=["spider", "fense", "vocab", "spider_fl"])
    print("Running aac_metrics.Evaluate ...")
    corpus_scores, results = evaluate(candidates, mult_references)
    print("Corpus-level scores:", {k: float(v) for k, v in corpus_scores.items()})

    fields = {
        "spider": results["spider"].tolist(),
        "spider_fl": results["spider_fl"].tolist(),
        "vocab": results["vocab.cands"].tolist(),
        "fense": results["fense"].tolist(),
        "sbert_sim": results["sbert_sim"].tolist(),
        "cider_d": results["cider_d"].tolist(),
        "spice": results["spice"].tolist(),
    }

    print("Setting results ...")
    for name, values in fields.items():
        by_id = dict(zip(ids, values))
        dataset.set_values(name, by_id, key_field="id")
        dataset.set_values(f"{args.pred_field}.{name}", by_id, key_field="id")

    print(f"\nDone. Wrote {list(fields.keys())} for {len(ids)} samples.")


if __name__ == "__main__":
    main()
