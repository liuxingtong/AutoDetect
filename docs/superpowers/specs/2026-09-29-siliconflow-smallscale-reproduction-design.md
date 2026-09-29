# AutoDetect SiliconFlow Small-Scale Reproduction Design

## Objective

Reproduce the central weakness-detection trend from *AutoDetect: Towards a
Unified Framework for Automated Weakness Detection in Large Language Models*
at substantially lower cost. The experiment must determine whether adaptive
question generation finds progressively lower-scoring responses and a
non-trivial identification success rate (ISR) on a target language model.

This first reproduction covers weakness detection only. It does not include
LoRA fine-tuning, downstream benchmark improvement, human annotation, or the
full mathematics and coding experiments.

## Constraints

- Store source code, virtual environments, cached data, datasets, logs, and
  outputs below `/data/liuxingtong` on the remote server.
- Do not commit credentials. Read the SiliconFlow API key from
  `/data/liuxingtong/secrets/siliconflow.env`, which must remain mode `0600`.
- Use `deepseek-ai/DeepSeek-V4-Flash` for the Examiner, Questioner, Assessor,
  reference-answer generator, and LLM judge.
- Use `Qwen/Qwen2.5-7B-Instruct` as the target model.
- Run all model inference through SiliconFlow's OpenAI-compatible API.
- Do not allocate a GPU for this phase.
- Keep the experiment near one tenth of the paper's full instruction-following
  search volume.
- Bound retries, API calls, output tokens, and wall-clock runtime so a malformed
  response or service outage cannot create an uncontrolled bill.
- Maintain a running record of reproduction problems and their resolutions.

## Experiment Scope

The reproduction uses four instruction-following subtasks chosen to cover the
paper's three top-level taxonomy groups:

| Top-level group | Subtask | Initial knowledge points |
| --- | --- | ---: |
| Format | Text format | 4 |
| Content | Word constraint | 2 |
| General constraints | Scenario simulation | 3 |
| General constraints | Emotional tone | 3 |

For each of the 12 initial knowledge points, the runner generates two seed
questions and performs three adaptive search iterations. This produces 60
target-model evaluations before any Assessor-discovered point. The paper's
published instruction-following taxonomy contains 37 knowledge points and uses
15 evaluations per point, or 555 evaluations, so the initial reproduction is
10.8% of that volume.

The Examiner receives the complete task description and official taxonomy, then
selects and normalizes the configured subset. This preserves comparability with
the released taxonomy while exercising the Examiner role. After each subtask,
the Assessor may propose at most one new knowledge point. A proposed point is
run only when it is materially different from the initial points and the global
sample cap has not been reached.

## System Design

### Configuration

A checked-in YAML configuration defines models, endpoint, selected subtasks,
seed count, adaptive iteration count, temperatures, token limits, retry limits,
random seed, sample cap, and output location. The API key is referenced only by
environment-variable name.

The default live profile uses:

- `base_url`: `https://api.siliconflow.cn/v1`
- `agent_model`: `deepseek-ai/DeepSeek-V4-Flash`
- `target_model`: `Qwen/Qwen2.5-7B-Instruct`
- `seed_questions_per_point`: `2`
- `adaptive_iterations_per_point`: `3`
- `max_initial_points`: `12`
- `max_discovered_points_per_subtask`: `1`
- `max_total_samples`: `80`
- `max_attempts_per_request`: `4`
- `request_timeout_seconds`: `120`

### Components

1. **SiliconFlow client**
   Wraps chat completions with timeouts, bounded exponential backoff, structured
   error messages, token accounting, and optional extraction of reasoning text.

2. **Structured-output parser**
   Accepts plain JSON and fenced JSON, validates required fields, and rejects
   malformed or semantically empty outputs. One repair request is allowed before
   the current sample is recorded as failed.

3. **Examiner**
   Selects a normalized taxonomy subset from the official task description and
   released taxonomy. The checked-in configuration remains the hard upper bound
   on scope.

4. **Questioner**
   Generates seed questions and adaptive questions. Adaptive generation receives
   a compact history containing prompts, knowledge points, scores, and failure
   explanations, prioritizing lower-scoring cases without repeating prompts.

5. **Target adapter**
   Sends each generated instruction to Qwen2.5-7B-Instruct with deterministic
   decoding where supported.

6. **Reference generator and judge**
   DeepSeek-V4-Flash generates a reference answer and applies the paper's
   reference-based 1–10 judge rubric. A score less than 4 is a discovered
   weakness. The parser requires a final numeric score in the inclusive range
   1–10.

7. **Assessor**
   Summarizes low-scoring cases for one subtask and may propose one distinct new
   knowledge point. The runner records both accepted and rejected proposals.

8. **Checkpoint and reporter**
   Appends every completed sample to JSONL and writes run metadata separately.
   It resumes by stable sample identifiers without issuing duplicate paid calls.
   The reporter produces CSV summaries, a Markdown report, and a PNG trend plot.

## Data Flow

1. Load configuration and validate the secret, models, limits, and output path.
2. Query the Examiner once and validate the selected taxonomy.
3. For every selected knowledge point, generate two seed questions.
4. For each question, obtain the target response, reference answer, and judge
   result; checkpoint the complete record immediately.
5. Rank history by score and generate one adaptive question per iteration,
   repeating evaluation and checkpointing for three iterations.
6. Ask the Assessor for at most one new point per subtask and record its decision.
7. Aggregate ISR, average score, and token usage overall, by subtask, by knowledge
   point, and by iteration.
8. Render the report and trend plot from checkpoints, without additional API
   calls.

## Evaluation and Success Criteria

The primary metric is the paper's ISR:

`ISR = count(score < 4) / count(valid judged samples)`

The reproduction is considered informative when all of the following hold:

- At least 90% of scheduled initial samples receive a valid score.
- The run produces at least one weakness (`score < 4`) or clearly documents that
  none were found within the reduced sample.
- The report compares seed and adaptive phases using mean score and ISR.
- The per-iteration plot and underlying CSV agree exactly.
- The conclusion distinguishes a directional reproduction from an exact
  numerical replication because the agent, judge, target model, and sample size
  differ from the paper.

A lower average score in later adaptive iterations is the expected paper-aligned
trend, but it is not hard-coded as a passing condition. A flat or rising trend is
reported as an experimental result rather than hidden or rerun selectively.

## Cost and Runtime Controls

- Run an offline test suite before any paid request.
- Run a one-knowledge-point live smoke test before the full experiment.
- Require an explicit CLI flag for a live run.
- Stop before scheduling a sample when `max_total_samples` is reached.
- Enforce per-request token limits and request timeouts.
- Retry only transient HTTP failures and rate limits; fail fast on authentication
  and invalid-model errors.
- Persist provider-reported token usage and the total request count.
- Support clean interruption and checkpoint-based resume.
- Use no local model weights and no GPU process.

## Testing and Verification

Tests use a scripted fake chat-completions transport and cover:

- plain and fenced JSON parsing;
- score extraction, including invalid and out-of-range scores;
- transient retry versus permanent failure behavior;
- stable sample identifiers and resume without duplicate calls;
- exact sample-cap enforcement;
- ISR and per-iteration aggregation;
- consistency between CSV data and plotted values;
- secret exclusion from generated artifacts and Git-tracked files.

After the offline suite passes, verification proceeds through a two-sample live
smoke run and then the capped full run. Final claims must cite fresh test output,
the run manifest, the number of valid samples, and the generated metrics.

## Reproduction Pitfall Log

`docs/reproduction-pitfalls.md` is a required, continuously updated artifact.
Each entry contains:

- date and experiment stage;
- observed symptom;
- root cause or best-supported diagnosis;
- resolution or workaround;
- impact on cost, runtime, comparability, and conclusions;
- prevention or follow-up action.

The initial entries will cover the official repository's hard-coded GPT and GPU
paths, unbounded retry loops, brittle JSON/score parsing, the released
`count_res.py` typo (`intend` instead of `indent`), DeepSeek reasoning text mixed
with final answers, the server's limited root-disk headroom, and slow direct Git
cloning from the server.

## Deliverables

- Reproducible Python package and CLI in the official AutoDetect checkout.
- Checked-in small-scale YAML configuration with no secrets.
- Automated offline tests.
- Raw JSONL checkpoints and immutable run manifest under
  `/data/liuxingtong/outputs/autodetect-smallscale/<run-id>/`.
- CSV metrics, PNG trend plot, and Markdown findings report.
- `docs/reproduction-pitfalls.md` with all encountered issues and resolutions.
- README instructions for fresh run, resume, and report-only regeneration.

## Explicit Non-Goals

- Exact reproduction of the paper's reported ISR values.
- Full three-domain evaluation in the first run.
- Training or fine-tuning a local model.
- Benchmarking post-training gains on IFEval, GSM8K, MATH, or HumanEval.
- Human validation of judge agreement.
- GPU allocation during the weakness-detection experiment.
