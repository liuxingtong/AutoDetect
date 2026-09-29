# AutoDetect SiliconFlow Small-Scale Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a bounded, resumable SiliconFlow reproduction of AutoDetect's instruction-following weakness-detection trend, including auditable metrics and a reproduction-pitfall log.

**Architecture:** Add a focused `autodetect_repro` package beside the released scripts instead of modifying their three duplicated monoliths. A typed configuration drives an OpenAI-compatible client, role-specific orchestration, append-only checkpoints, and offline reporting; paid live runs are gated by an explicit flag and hard caps.

**Tech Stack:** Python 3.8, `requests`, `PyYAML`, `matplotlib`, `pytest`, SiliconFlow OpenAI-compatible REST API.

**Spec:** `docs/superpowers/specs/2026-09-29-siliconflow-smallscale-reproduction-design.md`

## Global Constraints

- Keep source, virtual environments, caches, checkpoints, and outputs below `/data/liuxingtong` on the remote server.
- Read `SILICONFLOW_API_KEY` from `/data/liuxingtong/secrets/siliconflow.env`; never commit or print it.
- Use `deepseek-ai/DeepSeek-V4-Flash` for all AutoDetect roles, reference generation, and judging.
- Use `Qwen/Qwen2.5-7B-Instruct` as the target model.
- Use the SiliconFlow endpoint `https://api.siliconflow.cn/v1`.
- Schedule 2 seed questions and 3 adaptive questions for each of 12 initial knowledge points.
- Cap discovered points at 1 per subtask, total samples at 80, attempts per request at 4, and request timeout at 120 seconds.
- Require `--live` before any paid API call and run no local model weights or GPU process.
- Treat `score < 4` as a discovered weakness and preserve flat or adverse trends as legitimate results.
- Update `docs/reproduction-pitfalls.md` throughout implementation and experimentation.

## Review Focus

- A checkpoint ending in a partial or malformed JSONL line must resume from prior valid records and quarantine the bad line; Task 4 tests this.
- A provider response containing reasoning tags, duplicated final text, or fenced JSON must yield only the validated final payload; Task 2 tests this.
- An API response with HTTP 200 but no usable choice must consume a bounded retry and then fail with a redacted diagnostic; Task 2 tests this.
- A resumed run whose config differs from its manifest must stop before making a paid request; Task 4 tests this.
- An empty or partially valid experiment must still generate an honest report with denominators and missing-sample counts, never divide by zero; Task 5 tests this.

---

### Task 1: Reproduction Package, Typed Configuration, and Safe Defaults

**Files:**
- Create: `requirements-repro.txt`
- Create: `autodetect_repro/__init__.py`
- Create: `autodetect_repro/config.py`
- Create: `autodetect_repro/types.py`
- Create: `configs/siliconflow_smallscale.yaml`
- Create: `tests/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: YAML file path and process environment.
- Produces: `RunConfig`, `ModelConfig`, `SearchConfig`, `LimitsConfig`, `SubtaskConfig`, and `load_config(path: Path, environ: Mapping[str, str]) -> RunConfig` using Python-3.8-compatible `typing` annotations.

- [ ] **Step 1: Write failing configuration tests**

Add tests asserting the checked-in YAML loads the exact models and limits from Global Constraints, resolves output paths below `/data/liuxingtong`, raises `ConfigError` for a missing API-key variable, rejects non-positive limits, rejects duplicate knowledge points, and exposes no secret in `repr(config)`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_config.py`

Expected: collection fails because `autodetect_repro.config` does not exist.

- [ ] **Step 3: Implement configuration and immutable record types**

Implement frozen dataclasses in `types.py` and `load_config` in `config.py`. The YAML must select `text format`, `word constraint`, `scenario simulation`, and `emotional tone` with their 12 released knowledge points and reference the key only as `SILICONFLOW_API_KEY`.

- [ ] **Step 4: Add lightweight dependencies and ignore runtime artifacts**

Pin Python-3.8-compatible versions of `requests`, `PyYAML`, `matplotlib`, and `pytest` in `requirements-repro.txt`. Ignore `.venv/`, `__pycache__/`, `.pytest_cache/`, generated result directories, `.env`, and `*.env` without ignoring checked-in YAML.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `pytest -q tests/test_config.py`

Expected: all configuration tests pass.

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements-repro.txt autodetect_repro configs tests/test_config.py
git commit -m "feat: add safe reproduction configuration"
```

### Task 2: SiliconFlow Client and Structured Response Parsing

**Files:**
- Create: `autodetect_repro/client.py`
- Create: `autodetect_repro/parsing.py`
- Create: `tests/test_client.py`
- Create: `tests/test_parsing.py`

**Interfaces:**
- Consumes: `ModelConfig`, prompt messages, injectable `requests.Session`, and injectable sleep function.
- Produces: `ChatResult`, `SiliconFlowClient.complete(messages: Sequence[Message], *, temperature: float, max_tokens: int) -> ChatResult`, `parse_json_object(text: str) -> Dict[str, Any]`, and `parse_score(text: str) -> float`.

- [ ] **Step 1: Write failing parser tests**

Assert `parse_json_object` accepts plain and fenced JSON, extracts the last valid object after reasoning text, rejects missing required content, and handles `</think>` duplication. Assert `parse_score` accepts `[[3]]` and `Rating: [[3.5]]` but rejects absent, NaN, and values outside 1–10.

- [ ] **Step 2: Run parser tests and verify RED**

Run: `pytest -q tests/test_parsing.py`

Expected: import failure for `autodetect_repro.parsing`.

- [ ] **Step 3: Implement parsers**

Implement balanced-brace JSON candidate extraction and last-valid-candidate selection without `eval`. Strip reasoning sections only when delimiters are balanced, then validate scores as finite values in `[1, 10]`.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run: `pytest -q tests/test_parsing.py`

Expected: all parser tests pass.

- [ ] **Step 5: Write failing client tests**

Use a scripted fake session to assert request URL, bearer header presence without exposing its value in exceptions, model and token parameters, provider token accounting, retry of 429/503/timeouts with bounded backoff, immediate failure for 401/403/404, bounded failure for HTTP 200 without a usable choice, and exactly 4 total attempts.

- [ ] **Step 6: Run client tests and verify RED**

Run: `pytest -q tests/test_client.py`

Expected: import failure for `autodetect_repro.client`.

- [ ] **Step 7: Implement `SiliconFlowClient`**

Use `requests.Session.post(..., timeout=120)`, classify retryable failures, capture provider usage, and raise `ApiError` whose message includes status/model/request number but never headers, request bodies, or secrets.

- [ ] **Step 8: Run Task 2 tests and verify GREEN**

Run: `pytest -q tests/test_client.py tests/test_parsing.py`

Expected: all client and parser tests pass.

- [ ] **Step 9: Commit**

```bash
git add autodetect_repro/client.py autodetect_repro/parsing.py tests/test_client.py tests/test_parsing.py
git commit -m "feat: add bounded SiliconFlow client"
```

### Task 3: AutoDetect Roles and Prompt Contracts

**Files:**
- Create: `autodetect_repro/prompts.py`
- Create: `autodetect_repro/roles.py`
- Create: `tests/test_roles.py`

**Interfaces:**
- Consumes: `SiliconFlowClient`, `RunConfig`, official taxonomy, `SampleRecord` history.
- Produces: `Examiner.select_taxonomy() -> Tuple[KnowledgePoint, ...]`, `Questioner.generate_seeds(point: KnowledgePoint) -> Tuple[Question, ...]`, `Questioner.generate_adaptive(point: KnowledgePoint, history: Sequence[SampleRecord], iteration: int) -> Question`, `Evaluator.evaluate(question: Question) -> SampleRecord`, and `Assessor.propose(subtask: SubtaskConfig, history: Sequence[SampleRecord]) -> AssessorDecision`.

- [ ] **Step 1: Write failing role-contract tests**

With a fake client, assert the Examiner returns exactly the configured 12 initial points and cannot add unconfigured subtasks; seed generation returns exactly 2 distinct questions per point; adaptive prompts include compact sorted history and iteration number; evaluation calls target, reference, and judge models in that order; scores below 4 set `is_weakness=True`; and the Assessor accepts at most one distinct non-empty point per subtask.

- [ ] **Step 2: Run role tests and verify RED**

Run: `pytest -q tests/test_roles.py`

Expected: import failure for `autodetect_repro.roles`.

- [ ] **Step 3: Implement prompt builders**

Define one function per role in `prompts.py`, adapting the released AutoDetect prompts while requiring concise JSON fields. Keep the paper's reference-based 1–10 rubric and explicit rule that an incorrect answer cannot score above 3.

- [ ] **Step 4: Implement role classes**

Parse every structured response through Task 2 helpers. Allow one repair call for malformed structured output, reject duplicate prompts by normalized text, and keep all model calls behind the injected client.

- [ ] **Step 5: Run role tests and verify GREEN**

Run: `pytest -q tests/test_roles.py`

Expected: all role-contract tests pass.

- [ ] **Step 6: Commit**

```bash
git add autodetect_repro/prompts.py autodetect_repro/roles.py tests/test_roles.py
git commit -m "feat: implement AutoDetect role workflow"
```

### Task 4: Append-Only Checkpoints, Manifest Validation, and Resumable Runner

**Files:**
- Create: `autodetect_repro/checkpoint.py`
- Create: `autodetect_repro/runner.py`
- Create: `tests/test_checkpoint.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Consumes: `RunConfig`, role objects, run directory, optional existing manifest and JSONL checkpoint.
- Produces: `sample_id(point: KnowledgePoint, phase: str, iteration: int, ordinal: int) -> str`, `CheckpointStore`, `ExperimentRunner.run() -> RunSummary`, and immutable `manifest.json` plus `samples.jsonl`.

- [ ] **Step 1: Write failing checkpoint tests**

Assert stable identifiers across processes, atomic manifest creation, append-and-fsync behavior, valid-record loading, quarantine of a malformed final line, rejection of corruption in the middle of a file, and exclusion of secret values from manifest and samples.

- [ ] **Step 2: Run checkpoint tests and verify RED**

Run: `pytest -q tests/test_checkpoint.py`

Expected: import failure for `autodetect_repro.checkpoint`.

- [ ] **Step 3: Implement checkpoint storage**

Use canonical JSON plus SHA-256 for stable identifiers and config fingerprints. Write manifests through a temporary file plus `os.replace`; append one compact JSON object per line and flush/fsync before returning.

- [ ] **Step 4: Run checkpoint tests and verify GREEN**

Run: `pytest -q tests/test_checkpoint.py`

Expected: all checkpoint tests pass.

- [ ] **Step 5: Write failing runner tests**

Assert seed-before-adaptive ordering, exact skip of completed IDs on resume, no paid call when manifest fingerprint differs, immediate stop at 80 total samples, no more than one accepted discovered point per subtask, preservation of failed samples in denominators metadata, and clean stop after an injected interrupt.

- [ ] **Step 6: Run runner tests and verify RED**

Run: `pytest -q tests/test_runner.py`

Expected: import failure for `autodetect_repro.runner`.

- [ ] **Step 7: Implement the experiment runner**

The runner owns scheduling and caps; roles own prompts and calls. Persist each completed or failed sample immediately, derive resume state only from the manifest and checkpoint, and return counts for scheduled, valid, failed, weak, requests, and tokens.

- [ ] **Step 8: Run Task 4 tests and verify GREEN**

Run: `pytest -q tests/test_checkpoint.py tests/test_runner.py`

Expected: all checkpoint and runner tests pass.

- [ ] **Step 9: Commit**

```bash
git add autodetect_repro/checkpoint.py autodetect_repro/runner.py tests/test_checkpoint.py tests/test_runner.py
git commit -m "feat: add resumable capped experiment runner"
```

### Task 5: Metrics, CSV, Markdown Report, and Trend Plot

**Files:**
- Create: `autodetect_repro/reporting.py`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Consumes: `manifest.json` and `samples.jsonl` only; performs no model calls.
- Produces: `compute_metrics(records: Sequence[SampleRecord]) -> Metrics`, `write_report(run_dir: Path) -> ReportPaths`, `metrics.csv`, `iterations.csv`, `report.md`, and `score_by_iteration.png`.

- [ ] **Step 1: Write failing reporting tests**

Use fixed records to assert overall/category/point/phase ISR and mean score, `score < 4` boundary behavior, request/token totals, missing-sample counts, zero-valid-sample output without division by zero, CSV row values, and that plotting reads the exact rows written to `iterations.csv`.

- [ ] **Step 2: Run reporting tests and verify RED**

Run: `pytest -q tests/test_reporting.py`

Expected: import failure for `autodetect_repro.reporting`.

- [ ] **Step 3: Implement metrics and renderers**

Use the standard `csv` module and headless matplotlib `Agg`. `report.md` must state model substitutions, sample reduction, valid/failed denominators, whether adaptive mean score decreased, and that the result is directional rather than an exact numerical replication.

- [ ] **Step 4: Run reporting tests and verify GREEN**

Run: `pytest -q tests/test_reporting.py`

Expected: all reporting tests pass and a non-empty PNG is created under the pytest temporary directory.

- [ ] **Step 5: Commit**

```bash
git add autodetect_repro/reporting.py tests/test_reporting.py
git commit -m "feat: report AutoDetect reproduction metrics"
```

### Task 6: Safe CLI, Smoke Profile, and Report-Only Mode

**Files:**
- Create: `autodetect_repro/cli.py`
- Create: `autodetect_repro/__main__.py`
- Create: `configs/siliconflow_smoke.yaml`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `python -m autodetect_repro`, config path, run directory, environment, and CLI flags.
- Produces: `main(argv: Optional[Sequence[str]] = None) -> int` with `run`, `resume`, and `report` subcommands.

- [ ] **Step 1: Write failing CLI tests**

Assert `run` and `resume` refuse to construct a client without `--live`; `report` never requires a key; the smoke config schedules one point with two total samples; `--dry-run` prints models, counts, and paths but not secrets; invalid output paths outside `/data/liuxingtong` are rejected; and exit codes distinguish configuration, API, and incomplete-run failures.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `pytest -q tests/test_cli.py`

Expected: import failure for `autodetect_repro.cli`.

- [ ] **Step 3: Implement CLI entry points**

Wire Task 1 configuration, Task 2 client, Task 3 roles, Task 4 runner, and Task 5 reporter. Require `--live` for any network path and redact exception output before writing stderr or run metadata.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run: `pytest -q tests/test_cli.py`

Expected: all CLI tests pass.

- [ ] **Step 5: Run the complete offline suite**

Run: `pytest -q`

Expected: all tests pass with zero warnings caused by project code.

- [ ] **Step 6: Commit**

```bash
git add autodetect_repro/cli.py autodetect_repro/__main__.py configs/siliconflow_smoke.yaml tests/test_cli.py
git commit -m "feat: add guarded reproduction CLI"
```

### Task 7: Runbook, Secret Audit, and Reproduction Pitfall Log

**Files:**
- Create: `docs/reproduction-pitfalls.md`
- Create: `docs/siliconflow-reproduction-runbook.md`
- Modify: `README.md`
- Create: `tests/test_repository_safety.py`

**Interfaces:**
- Consumes: repository tree and the CLI from Task 6.
- Produces: operator instructions, required pitfall entries, and a tracked-file secret-safety check.

- [ ] **Step 1: Write failing repository-safety tests**

Assert every Git-tracked text file excludes high-entropy credentials matching `sk-[A-Za-z0-9]{20,}` and the literal provided key; required docs exist; the pitfall table has symptom, cause, resolution, impact, and prevention columns; and the runbook uses only `/data/liuxingtong` for environments, caches, and outputs.

- [ ] **Step 2: Run safety tests and verify RED**

Run: `pytest -q tests/test_repository_safety.py`

Expected: failure because the required documentation does not exist.

- [ ] **Step 3: Write the runbook and README entry**

Document virtual-environment creation under `/data/liuxingtong/envs`, cache environment variables under `/data/liuxingtong/cache`, offline tests, dry run, live smoke, full live run, resume, and report regeneration.

- [ ] **Step 4: Create and seed the pitfall log**

Record the released scripts' hard-coded models/GPU, 50-attempt loops, brittle fenced-JSON and score parsing, `intend=4` typo, DeepSeek reasoning/final mixing, low root-disk headroom, and stalled server-to-GitHub clone. Mark observed versus code-inspection findings explicitly.

- [ ] **Step 5: Run safety and full tests and verify GREEN**

Run: `pytest -q tests/test_repository_safety.py && pytest -q`

Expected: all tests pass and no tracked secret pattern is reported.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/reproduction-pitfalls.md docs/siliconflow-reproduction-runbook.md tests/test_repository_safety.py
git commit -m "docs: add reproduction runbook and pitfall log"
```

### Task 8: Remote Environment, Live Smoke, Capped Experiment, and Final Evidence

**Files:**
- Modify as issues arise: `docs/reproduction-pitfalls.md`
- Generate outside Git: `/data/liuxingtong/outputs/autodetect-smallscale/<run-id>/manifest.json`
- Generate outside Git: `/data/liuxingtong/outputs/autodetect-smallscale/<run-id>/samples.jsonl`
- Generate outside Git: `/data/liuxingtong/outputs/autodetect-smallscale/<run-id>/{metrics.csv,iterations.csv,report.md,score_by_iteration.png}`

**Interfaces:**
- Consumes: committed package, configs, `/data/liuxingtong/secrets/siliconflow.env`, and SiliconFlow service.
- Produces: fresh verification output and the final reproducibility artifacts.

- [ ] **Step 1: Synchronize the reviewed commits to the server checkout**

Run locally: `rsync -az --exclude '.DS_Store' ./ liuxingtong@115.182.62.246:/data/liuxingtong/src/AutoDetect/`

Expected: remote `git status --porcelain` is empty and `git rev-parse HEAD` matches local HEAD.

- [ ] **Step 2: Create the isolated CPU environment under `/data`**

Set `PIP_CACHE_DIR=/data/liuxingtong/cache/pip`, then run remotely: `python3 -m venv /data/liuxingtong/envs/autodetect-repro && /data/liuxingtong/envs/autodetect-repro/bin/pip install -r /data/liuxingtong/src/AutoDetect/requirements-repro.txt`

Expected: installation exits 0; no environment or package cache is created under `/home/liuxingtong`.

- [ ] **Step 3: Run the complete offline suite on the server**

Run remotely: `cd /data/liuxingtong/src/AutoDetect && /data/liuxingtong/envs/autodetect-repro/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Run and inspect the live smoke experiment**

Load the mode-0600 secret file, set cache and temporary directories below `/data/liuxingtong`, then run `python -m autodetect_repro run --config configs/siliconflow_smoke.yaml --live`.

Expected: exactly two valid or explicitly failed sample records, bounded request counts, report artifacts, no secret in logs, and no GPU process owned by `liuxingtong`.

- [ ] **Step 5: Update and commit pitfalls discovered by the smoke run**

Add only evidence-backed entries or amendments, run `pytest -q tests/test_repository_safety.py`, and commit with `docs: record SiliconFlow smoke-test pitfalls` when the file changed.

- [ ] **Step 6: Run the capped full experiment and resume if interrupted**

Run remotely: `python -m autodetect_repro run --config configs/siliconflow_smallscale.yaml --live`; use the printed run directory with the `resume` command only after verifying the manifest fingerprint.

Expected: no more than 80 samples, at least 90% valid initial samples or a documented provider failure, immediate checkpoints, and no GPU process owned by `liuxingtong`.

- [ ] **Step 7: Regenerate the final report without network access**

Run remotely: `python -m autodetect_repro report --run-dir <absolute-run-directory>` with `SILICONFLOW_API_KEY` unset.

Expected: report regeneration exits 0 and makes no HTTP requests; CSV and PNG timestamps update from existing checkpoint data.

- [ ] **Step 8: Run final verification and capture evidence**

Run remotely: the full `pytest -q`, `git status --porcelain`, the repository-safety test's high-entropy credential scan, artifact existence checks, JSONL valid-record count, sample cap assertion, and `nvidia-smi` process query.

Expected: all tests pass, Git is clean, secret grep has no matches, required artifacts are non-empty, sample count is at most 80, and no GPU process belongs to `liuxingtong`.

- [ ] **Step 9: Commit final evidence-backed documentation changes**

```bash
git add docs/reproduction-pitfalls.md README.md docs/siliconflow-reproduction-runbook.md
git commit -m "docs: finalize AutoDetect reproduction findings"
```

Skip the commit only when those tracked files are unchanged; raw experiment outputs remain outside Git.
