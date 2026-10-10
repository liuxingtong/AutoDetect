# AutoDetect Adaptive Framework Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the adaptive AutoDetect runtime so controller decisions, novelty filtering, behavioral branches, promotion proposals, checkpoint recovery, and reproducible deployment operate as one tested end-to-end framework.

**Architecture:** Keep the existing baseline runner unchanged behind `search.mode=baseline`. Adaptive modes use typed decisions and events around the existing Examiner, Questioner, Target, Evaluator, and Assessor roles; all mutable adaptive state is reconstructible from append-only events, while taxonomy mutation remains disabled unless explicitly configured.

**Tech Stack:** Python 3.9, dataclasses, JSONL checkpoints, PyYAML, requests, pytest, optional FastAPI/Uvicorn/Transformers embedding service.

**Spec:** `docs/superpowers/specs/2026-10-10-adaptive-exploration-design.md`

## Global Constraints

- `baseline` must preserve its current scheduling behavior and old `samples.jsonl` compatibility.
- Every behavior change starts with a failing test and ends with the full test suite passing.
- Every task is committed independently; generated experiment outputs are not included in code commits.
- LLM output never directly changes budgets, verified evidence, or taxonomy without runtime validation.
- `search.auto_promote` defaults to `false`; promotion proposals are persisted even when automatic taxonomy mutation is disabled.
- Total Target calls never exceed `limits.max_total_samples` or the current decision's test budget.

## Review Focus

- Resume after interruption must reconstruct the same branch, behavior-cluster, and target-call state as an uninterrupted run.
- Empty, malformed, duplicate, and uncertain candidates must have explicit recorded outcomes rather than disappearing silently.
- Controller budgets of zero, oversized budgets, and remaining-budget exhaustion must be clamped deterministically.
- Event append failure after sample persistence must stop the round without losing the saved sample or silently continuing with inconsistent state.
- Existing manifests created before embedding configuration existed must remain resumable when embedding is disabled.

---

### Task 1: Stabilize Embedding Configuration and Backward Compatibility

**Files:**
- Modify: `autodetect_repro/types.py`
- Modify: `autodetect_repro/config.py`
- Modify: `autodetect_repro/checkpoint.py`
- Modify: `autodetect_repro/cli.py`
- Modify: `requirements-repro.txt`
- Test: `tests/test_config.py`
- Test: `tests/test_checkpoint.py`
- Create: `tests/test_embedding_client.py`

**Interfaces:**
- Consumes: optional YAML `embedding` mapping and legacy manifests.
- Produces: validated `EmbeddingConfig`, compatible fingerprints, and a callable embedding provider.

- [ ] Add failing tests for legacy manifest resume, propagation of local embedding options, remote response cardinality validation, and missing optional server dependencies.
- [ ] Run the focused tests and confirm the new cases fail for the expected reasons.
- [ ] Implement canonical public-config compatibility, complete option propagation, strict remote response validation, and documented optional dependencies.
- [ ] Run focused tests, then `pytest -q` and `scripts/test_minimal_stack.py`.
- [ ] Commit as `fix: stabilize embedding configuration and recovery`.

### Task 2: Make Controller Decisions Drive Candidate Generation and Test Budgets

**Files:**
- Modify: `autodetect_repro/types.py`
- Modify: `autodetect_repro/prompts.py`
- Modify: `autodetect_repro/roles.py`
- Modify: `autodetect_repro/runner.py`
- Test: `tests/test_adaptive_prompts.py`
- Test: `tests/test_roles.py`
- Test: `tests/test_runner_novelty.py`

**Interfaces:**
- Consumes: validated `SearchDecision`, branch context, history, and remaining global budget.
- Produces: typed candidates generated for the decision objective and up to `test_budget` evaluated samples.

- [ ] Add failing tests proving strategy/objective reach the Questioner and that `test_budget` controls selected/evaluated candidates without exceeding the global cap.
- [ ] Run focused tests and confirm failures identify the current fixed-one-sample path.
- [ ] Introduce the typed candidate contract and pass the controller decision into adaptive generation.
- [ ] Schedule selected candidates with stable ordinals and deterministic sample IDs up to the clamped test budget.
- [ ] Run focused and full tests.
- [ ] Commit as `feat: apply adaptive decisions to generation and budgets`.

### Task 3: Complete Hybrid Novelty Classification and Exploration Selection

**Files:**
- Modify: `autodetect_repro/novelty.py`
- Modify: `autodetect_repro/prompts.py`
- Modify: `autodetect_repro/roles.py`
- Modify: `autodetect_repro/cli.py`
- Modify: `autodetect_repro/runner.py`
- Test: `tests/test_novelty.py`
- Test: `tests/test_roles.py`
- Test: `tests/test_runner_novelty.py`

**Interfaces:**
- Consumes: candidates, a frozen history snapshot, embedding Top-K neighbors, objective, and random exploration ratio.
- Produces: typed keep/reject/uncertain decisions and a deterministic budget-bounded selection.

- [ ] Add failing tests for normalized duplicates, LLM Duplicate/Related/Distinct/Uncertain parsing, classifier failure, uncertain quota, and seeded random exploration.
- [ ] Implement an LLM novelty classifier and wire it through CLI construction.
- [ ] Implement explicit hard-filter reasons, uncertain preservation, and seeded exploration selection.
- [ ] Persist complete candidate-filter events without secrets.
- [ ] Run focused and full tests.
- [ ] Commit as `feat: connect hybrid novelty classification`.

### Task 4: Recover Adaptive State from Events

**Files:**
- Modify: `autodetect_repro/checkpoint.py`
- Modify: `autodetect_repro/behavior.py`
- Modify: `autodetect_repro/branches.py`
- Modify: `autodetect_repro/analysis_pipeline.py`
- Modify: `autodetect_repro/runner.py`
- Test: `tests/test_checkpoint.py`
- Test: `tests/test_behavior.py`
- Test: `tests/test_branches.py`
- Test: `tests/test_analysis_pipeline.py`
- Test: `tests/test_runner_novelty.py`

**Interfaces:**
- Consumes: legacy or current `events.jsonl` plus persisted samples.
- Produces: reconstructed behavioral clusters, branches, counters, and controller statistics.

- [ ] Add failing equivalence tests comparing uninterrupted execution with interrupted/resumed execution.
- [ ] Add event schema/version validation and recoverable handling of a truncated final event line.
- [ ] Implement replay methods for behavior clusters, branches, and analysis counters.
- [ ] Make Runner replay events before scheduling additional work.
- [ ] Run focused and full tests.
- [ ] Commit as `feat: restore adaptive state from event history`.

### Task 5: Connect Verification, Counterexamples, and Promotion Proposals

**Files:**
- Modify: `autodetect_repro/types.py`
- Modify: `autodetect_repro/branches.py`
- Modify: `autodetect_repro/analysis_pipeline.py`
- Modify: `autodetect_repro/roles.py`
- Modify: `autodetect_repro/runner.py`
- Modify: `autodetect_repro/cli.py`
- Test: `tests/test_branches.py`
- Test: `tests/test_analysis_pipeline.py`
- Test: `tests/test_runner_novelty.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: supporting failures, successful controls/counterexamples, verification samples, and Examiner semantic judgment.
- Produces: evidence-gated `PromotionProposal` events and optional explicit taxonomy mutation.

- [ ] Add failing state-transition tests for candidate, verifying, promoted, and rejected branches.
- [ ] Record counterexamples and controlled verification results against the intended branch.
- [ ] Invoke promotion judgment only after deterministic evidence gates pass.
- [ ] Persist proposals; keep taxonomy unchanged when `auto_promote=false` and test explicit mutation when true.
- [ ] Run focused and full tests.
- [ ] Commit as `feat: complete branch verification and promotion flow`.

### Task 6: Add End-to-End Offline Acceptance and Reporting

**Files:**
- Modify: `autodetect_repro/reporting.py`
- Modify: `autodetect_repro/cli.py`
- Create: `tests/test_adaptive_end_to_end.py`
- Modify: `tests/test_reporting.py`
- Modify: `docs/siliconflow-reproduction-runbook.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: deterministic fake-role runs for baseline, adaptive-rule, and adaptive-LLM modes plus saved samples/events.
- Produces: comparable semantic/behavioral metrics and a documented operating procedure.

- [ ] Add an offline end-to-end test for all three modes, interruption/resume, budget equality, and report regeneration.
- [ ] Extend reports with unique question gain, behavioral richness/gain, discovery rate, branch status, and promotion proposals.
- [ ] Document embedding startup, health checks, dependency installation, run/resume, and failure recovery.
- [ ] Run `pytest -q`, `compileall`, `scripts/test_minimal_stack.py`, and `git diff --check`.
- [ ] Commit as `test: verify adaptive framework end to end`.

### Task 7: Final Review and Release Readiness

**Files:**
- Modify only files required by review findings.

- [ ] Review the complete diff against every acceptance item in the design spec.
- [ ] Resolve all critical and important review findings with regression tests.
- [ ] Run the complete verification suite again from a clean process.
- [ ] Confirm `git status` contains only intentionally untracked experiment artifacts.
- [ ] Commit any review fixes as a focused final commit.
