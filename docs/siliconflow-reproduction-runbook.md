# SiliconFlow Small-Scale Reproduction Runbook

All commands assume the repository is located at
`/data/liuxingtong/src/AutoDetect`.

## 1. Prepare CPU-only paths and environment

```bash
export PIP_CACHE_DIR=/data/liuxingtong/cache/pip
export HF_HOME=/data/liuxingtong/cache/huggingface
export MPLCONFIGDIR=/data/liuxingtong/cache/matplotlib
export TMPDIR=/data/liuxingtong/tmp
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" "$TMPDIR"
python3 -m venv /data/liuxingtong/envs/autodetect-repro
/data/liuxingtong/envs/autodetect-repro/bin/pip install -r /data/liuxingtong/src/AutoDetect/requirements-repro.txt
```

The reproduction uses remote APIs and must not start a local model or GPU
process.

## 2. Load the credential without printing it

```bash
test "$(stat -c '%a' /data/liuxingtong/secrets/siliconflow.env)" = "600"
set -a
source /data/liuxingtong/secrets/siliconflow.env
set +a
test -n "$SILICONFLOW_API_KEY"
```

Do not enable shell tracing while the secret is loaded.

## 3. Run offline verification

```bash
cd /data/liuxingtong/src/AutoDetect
/data/liuxingtong/envs/autodetect-repro/bin/pytest -q
```

No API request is made by the test suite.

## 4. Inspect the full profile without network calls

```bash
cd /data/liuxingtong/src/AutoDetect
/data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro run \
  --config configs/siliconflow_smallscale.yaml \
  --run-dir /data/liuxingtong/outputs/autodetect-smallscale/dry-run \
  --dry-run
```

Confirm that the output names 60 planned samples for the 12 initial knowledge
points, an 80-sample cap that allows four discovered points, the expected two
models, and `network_calls: false`. A complete run can therefore contain 80
samples: 60 from initial points plus 20 from discovered points.

## 5. Run the two-sample live smoke

```bash
cd /data/liuxingtong/src/AutoDetect
REPRO_SMOKE_ID=smoke-$(date -u +%Y%m%dT%H%M%SZ)
/data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro run \
  --config configs/siliconflow_smoke.yaml \
  --run-dir "/data/liuxingtong/outputs/autodetect-smallscale/$REPRO_SMOKE_ID" \
  --live
```

Expect exactly two checkpoint records. Inspect `report.md`, `samples.jsonl`, and
the process list before continuing.

## 6. Run the capped experiment

```bash
cd /data/liuxingtong/src/AutoDetect
REPRO_RUN_ID=full-$(date -u +%Y%m%dT%H%M%SZ)
REPRO_RUN_DIR="/data/liuxingtong/outputs/autodetect-smallscale/$REPRO_RUN_ID"
/data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro run \
  --config configs/siliconflow_smallscale.yaml \
  --run-dir "$REPRO_RUN_DIR" \
  --live
```

The runner stops before sample 81 and fsyncs each completed or failed sample.
Exit code 4 means the run is interrupted or has fewer than 90% valid initial
samples; it does not mean existing checkpoints were discarded.

## 7. Resume the same manifest

```bash
cd /data/liuxingtong/src/AutoDetect
/data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro resume \
  --config configs/siliconflow_smallscale.yaml \
  --run-dir "$REPRO_RUN_DIR" \
  --live
```

A different configuration fingerprint is rejected before a model call. Keep the
same configuration and run directory when resuming.

## 8. Regenerate reports with the key absent

```bash
cd /data/liuxingtong/src/AutoDetect
env -u SILICONFLOW_API_KEY \
  /data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro report \
  --run-dir "$REPRO_RUN_DIR"
```

This recreates `metrics.csv`, `iterations.csv`, `report.md`, and
`score_by_iteration.png` from the manifest and checkpoint only.

## 9. Verify resource and artifact boundaries

```bash
df -h /data
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
find "$REPRO_RUN_DIR" -maxdepth 1 -type f -size +0 -print
wc -l "$REPRO_RUN_DIR/samples.jsonl"
```

No GPU process should belong to this reproduction. Update
`docs/reproduction-pitfalls.md` with any observed deviation before interpreting
the final trend.

## 10. Optional: enable adaptive search with a local embedding service

The default config uses `search.mode=baseline` and the embedding service is
disabled.  To exercise the adaptive path (controller + novelty filter +
LLM novelty classifier + branch/promotion flow) follow these steps.

### 10.1 Start the embedding service

```bash
/data/liuxingtong/envs/autodetect-repro/bin/pip install fastapi uvicorn numpy 'sentence-transformers>=2.2.0'
bash scripts/start_embedding_server.sh \
    --port 8080 \
    --model sentence-transformers/paraphrase-MiniLM-L6-v2
```

The server is health-checked at `http://localhost:8080/health`; a 200 OK means
it is ready.  If the model weights are not in `HF_HOME`, the server will
download them on first start.

### 10.2 Use the adaptive example config

```bash
REPRO_ADAPTIVE_RUN_DIR="$REPRO_RUN_DIR/adaptive"
/data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro run \
    --live \
    --config configs/config-with-embedding.example.yaml \
    --run-dir "$REPRO_ADAPTIVE_RUN_DIR"
```

The adaptive config sets `search.mode=adaptive_rule`, wires the local
embedding server as the novelty provider, and turns on `auto_promote=true`
so the branch verification + promotion flow runs end-to-end.

### 10.3 Resume after interruption

```bash
/data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro resume \
    --run-dir "$REPRO_ADAPTIVE_RUN_DIR"
```

The runner replays `events.jsonl` to restore the behavioral analyzer,
branch state, and target-call counter so the resumed run issues the
same search decisions as an uninterrupted run.  The new `## Branch &
Promotion` and `## Behavioral Diversity` sections in `report.md`
reflect the final branch status, promotion proposals, and unique-gain
metrics from the persisted events.

## 11. Recovery: detect and recover from a corrupted resume

The checkpoint store tolerates the final truncated line of `samples.jsonl`
but cannot recover mid-file corruption.  If `autodetect_repro resume` reports
a `CheckpointError` about line N, run:

```bash
# Inspect the samples that were kept.
wc -l "$REPRO_RUN_DIR/samples.jsonl" "$REPRO_RUN_DIR/quarantine.jsonl"
# Move the quarantined bytes aside and retry the resume.
mv "$REPRO_RUN_DIR/quarantine.jsonl" "$REPRO_RUN_DIR/quarantine.jsonl.recovered"
/data/liuxingtong/envs/autodetect-repro/bin/python -m autodetect_repro resume \
    --run-dir "$REPRO_RUN_DIR"
```

If a manifest fingerprint mismatch appears (e.g. config was edited mid-run),
the resume is refused to prevent silent state corruption.  Fix the config
back to the original or move the run to a fresh `--run-dir` and start over.
