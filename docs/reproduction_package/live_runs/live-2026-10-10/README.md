# Live run: 2026-10-10

This directory contains the artifacts of a real adaptive run against
the live siliconflow API on 2026-10-10.  It is the cleanest live-run
reproduction the project has produced to date, and is the reference
output that future refactors should be measured against.

## How this run was produced

```bash
export SILICONFLOW_API_KEY=sk-...your-key...
export AUTODETECT_OUTPUT_ROOT=/tmp/autodetect-live
python -m autodetect_repro run --live \
    --config $(pwd)/config.yaml \
    --run-dir /tmp/autodetect-live/run-3
```

The config is the same `config.yaml` in this directory; it is a
deliberately small-scale config (12 sample budget, 2 subtasks, 3
knowledge points) so the run finishes in about 5 minutes.

## Result summary

- 12 scheduled / 12 valid / 5 weak (ISR 41.67%)
- Mean score 5.33
- 3 branches discovered, 2 of them over the word-constraint point
- 10 `promotion_proposal` events (all `continue_verification`; the
  budget is too small to PROMOTE within 12 samples, which is the
  intended behavior — the gate is conservative by design)
- 7 `search_decision` events with strategy transitions
  explore → verify → exploit as failure signals accumulated
- 3 `behavior_cluster_update` events (label = "uncertain", because
  the LLM-judge calibration path is not enabled in this small run)
- 3 `failure_analysis` events with real LLM-generated signatures
  (e.g. "Use of forbidden word 'but' in the response")

## Files in this directory

| file | what it is |
| --- | --- |
| `config.yaml` | the exact config used for the run |
| `manifest.json` | run-start snapshot: config fingerprint + timestamp |
| `samples.jsonl` | one JSON object per evaluated sample |
| `events.jsonl` | one JSON object per pipeline event (branch / search / cluster / promotion) |
| `iterations.csv` | per-(phase, iteration) aggregates: total, valid, weak, ISR, mean score |
| `metrics.csv` | per-(level, key) breakdown (overall / category / subtask / point / phase) |
| `report.md` | human-readable summary; same content as `docs/reproduction_package/.../report.md` |
| `score_by_iteration.png` | 1280x720 bar chart of mean score and weak count per iteration |

## Bugs found and fixed during this run

Two bugs surfaced during the live run and were fixed in the same
session:

1. `embedding.enabled: false` was silently ignored by the config
   loader; the runner built a NoveltyFilter that called the
   siliconflow /v1/embeddings endpoint with an empty `input`,
   producing a 400 on every adaptive round.  Fixed in
   `autodetect_repro/config.py` (one-line change) and pinned by
   `tests/test_embedding_flag.py`.  Without this fix, this run
   would have shipped with 2 of 4 samples marked as "failed"
   for the wrong reason.

2. The default `configs/siliconflow_smallscale.yaml` was
   `search.mode: baseline` (the field was missing and the default
   is baseline).  The whole adaptive framework was wired but
   never triggered.  Fixed by adding `mode: adaptive_rule` to
   the config; pinned by `tests/test_default_config.py`.
