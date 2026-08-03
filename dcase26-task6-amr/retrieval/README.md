# Track B (query-similarity threshold) results — Clotho-Moment validation split

DCASE'26 Task 6 moment retrieval. `4918` ground-truth queries, validation split. Predictions: relative-peak-threshold merge over 10s/1s-hop sliding-window embeddings (see `retrieval/moment_retrieval.py`).

| Backend | R1@0.5 | R1@0.7 | mAP (ActivityNet, classwise) | queries covered |
|---|---|---|---|---|
| clap | 0.7391 | 0.4455 | 0.3973 | 4918/4918 |
| msclap | 0.7308 | 0.3845 | 0.3513 | 4918/4918 |

mAP uses FiftyOne's native `evaluate_detections(method="activitynet", classwise=True, compute_mAP=True)` over `predictions_<backend>` vs `ground_truth` on the `clotho-moment-video` dataset; `classwise=True` + per-query `str(qid)` labels ensure a prediction only matches the ground truth for the SAME query.
