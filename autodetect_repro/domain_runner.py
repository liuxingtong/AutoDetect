"""Fixed-condition math/code ablation adapter.

This module is intentionally *not* the formal AutoDetect experiment entry
point.  It keeps a small, fixed taxonomy and budget so novelty ablations can
compare runs without Assessor expansion, controller scheduling, branch
promotion, or taxonomy mutation.  Formal experiments must use
``python -m autodetect_repro.cli run --config ...`` instead.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .client import SiliconFlowClient
from .config import load_config
from .diversity import (
    behavioral_metrics,
)
from .parsing import parse_json_object, parse_score


DOMAIN_POINTS = {
    "math": (
        "algebra / arithmetic / basic_operations",
        "algebra / equations_and_inequalities / quadratic_equations",
        "geometry / trigonometry / trigonometric_identities",
        "analysis / derivatives / chain_rule",
        "algebra / polynomials / factorization_and_roots",
        "algebra / functions / composition_and_inverses",
        "geometry / coords_and_shapes / analytic_geometry",
        "analysis / integrals / substitution",
        "number_theory / modular_arithmetic / congruences",
        "probability / conditional_probability / bayes_rule",
    ),
    "code": (
        "string_processing / regular_expressions",
        "data_container_operations / filtering_and_selection",
        "logic_and_control_flow / exception_handling",
        "mathematics_and_algorithms / algorithm_design",
        "string_processing / parsing / tokenization",
        "data_structures / graphs / traversal",
        "data_structures / heaps / priority_queues",
        "file_handling / csv / robust_io",
        "functions_and_modules / decorators / state",
        "mathematics_and_algorithms / dynamic_programming / memoization",
    ),
}


def _call(client: SiliconFlowClient, prompt: str, temperature: float = 0.0):
    return client.complete(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )


def _json_call(client: SiliconFlowClient, prompt: str) -> Dict[str, Any]:
    last = ""
    for attempt in range(2):
        current = prompt if attempt == 0 else (
            "Return ONLY one valid JSON object. Do not use markdown fences or explanations.\n"
            "Original request:\n" + prompt + "\nPrevious invalid response:\n" + last[:2000]
        )
        result = _call(client, current, temperature=0.0)
        last = result.content
        try:
            return parse_json_object(last)
        except Exception:
            continue
    raise RuntimeError("structured generation failed after one repair")


def _domain_context(domain: str) -> str:
    if domain == "math":
        return "Solve a high-school or early-college mathematics problem. Correctness and reasoning matter."
    return "Write correct executable Python code for the requested programming problem. Prefer a self-contained function and include no unverifiable claims."


def _seed_prompt(domain: str, point: str) -> str:
    return (
        "You are generating evaluation questions for an LLM.\n"
        f"Domain: {domain}. Knowledge point: {point}.\n"
        f"{_domain_context(domain)}\n"
        "Return JSON only with key questions, a list of exactly two objects. "
        "Each object must have prompt and key_point. Make the two questions distinct, self-contained, and objectively checkable."
    )


def _adaptive_prompt(domain: str, point: str, history: Sequence[Dict[str, Any]]) -> str:
    compact = "\n".join(
        "- score={}: {}".format(item["score"], item["prompt"][:500])
        for item in sorted(history, key=lambda x: x["score"])[:4]
    )
    return (
        "You are an adaptive weakness-search question generator.\n"
        f"Domain: {domain}. Knowledge point: {point}.\n"
        f"{_domain_context(domain)}\n"
        "Generate one new, harder but still objectively checkable question. Avoid copying the previous questions. "
        "Return JSON only with keys prompt and key_point.\nPrevious search history:\n" + compact
    )


def _adaptive_candidates_prompt(
    domain: str, point: str, history: Sequence[Dict[str, Any]], count: int
) -> str:
    """Prompt the agent to emit N candidate adaptive questions in one JSON object."""
    compact = "\n".join(
        "- score={}: {}".format(item["score"], item["prompt"][:500])
        for item in sorted(history, key=lambda x: x["score"])[:4]
    )
    return (
        "You are an adaptive weakness-search question generator.\n"
        f"Domain: {domain}. Knowledge point: {point}.\n"
        f"{_domain_context(domain)}\n"
        f"Generate {count} NEW, harder but still objectively checkable questions. "
        "Avoid copying the previous questions. Each candidate should explore a slightly "
        "different angle (boundary case, edge constraint, unusual combination, etc.). "
        "Return JSON only with key 'questions', a list of exactly "
        f"{count} objects. Each object must have 'prompt' and 'key_point'.\n"
        "Previous search history:\n" + compact
    )


def _novelty_judge_prompt(candidate: str, neighbors: Sequence[str]) -> str:
    """Ask the LLM to label a candidate as duplicate/related/distinct.

    Mirrors the rubric used by NoveltyFilter; intended to run on the same agent
    model (DeepSeek) so it can be consumed by the same calibration pipeline.
    """
    if not neighbors:
        return (
            "Decide if the following evaluation question is essentially identical "
            "to any of the previous questions (DUPLICATE), shares most of its setup "
            "but is meaningfully different (RELATED), or is a clearly distinct "
            "question (DISTINCT). Return JSON only with key 'label'.\n\n"
            f"Question:\n{candidate}"
        )
    joined = "\n".join(f"- {text[:500]}" for text in neighbors)
    return (
        "Decide if the candidate evaluation question is essentially identical to "
        "any of the previous questions (DUPLICATE), shares most of its setup but is "
        "meaningfully different (RELATED), or is a clearly distinct question "
        "(DISTINCT). Return JSON only with key 'label'.\n\n"
        f"Previous questions:\n{joined}\n\nCandidate:\n{candidate}"
    )


# Tiny stopword set so semantic_novelty is a meaningful per-word Jaccard distance
# instead of a raw overlap on "the", "a", "Write", etc.
_STOPWORDS = frozenset(
    """a an the of to for in on at by with and or if is are be do does did
    write your function returns return takes following should must given let
    when where what which that this these those using used use from as into
    it its their there here all any some more most much many one two three
    show step steps how why can could may might will would has have had not
    no nor only own same so than then too very s t n p q r x y z i j k
    """.split()
)


def _normalize_tokens(text: str) -> Tuple[str, ...]:
    lowered = re.sub(r"[^a-zA-Z0-9_]+", " ", text.lower())
    return tuple(
        token for token in lowered.split() if token and token not in _STOPWORDS
    )


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 0.0
    l_set, r_set = set(left), set(right)
    if not l_set or not r_set:
        return 0.0
    return len(l_set & r_set) / len(l_set | r_set)


def _parse_novelty_label(text: str) -> str:
    try:
        payload = parse_json_object(text)
    except Exception:
        lowered = text.lower()
        for label in ("duplicate", "related", "distinct"):
            if label in lowered:
                return label
        return "distinct"
    raw = str(payload.get("label", "")).strip().lower()
    if raw not in {"duplicate", "related", "distinct"}:
        return "distinct"
    return raw


def _emit_event(events_path: Path, event_type: str, payload: Dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"event_type": event_type, **payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


def _reference_prompt(domain: str, prompt: str) -> str:
    return (
        "Produce a correct reference answer for the following evaluation question. "
        f"This is a {domain} task. Give the answer directly and include enough reasoning or code to verify correctness.\n\n"
        + prompt
    )


def _judge_prompt(domain: str, prompt: str, answer: str, reference: str) -> str:
    return (
        "Act as an impartial judge. Score the assistant answer from 1 to 10. "
        "For math, prioritize correctness; for code, prioritize executable correctness and adherence to the request. "
        "Use score below 4 for a substantive failure. End exactly with Rating: [[number]].\n\n"
        f"Domain: {domain}\nQuestion:\n{prompt}\n\nReference:\n{reference}\n\nAssistant:\n{answer}"
    )


def _flatten(raw: Any, prefix: str = "") -> List[str]:
    if isinstance(raw, dict):
        values = []
        for key, value in raw.items():
            next_prefix = f"{prefix} / {key}" if prefix else str(key)
            values.extend(_flatten(value, next_prefix))
        return values
    return [prefix] if prefix else []


def run(
    domain: str,
    config_path: Path,
    run_dir: Path,
    *,
    candidates_per_round: int = 1,
    use_novelty_filter: bool = True,
    novelty_top_k: int = 1,
) -> Dict[str, Any]:
    env = __import__("os").environ
    config = load_config(config_path, env)
    # Use a higher retry count for all three roles so that a single
    # transient SiliconFlow glitch never drops a sample.  4 attempts cover
    # the worst case: attempt-1 transport error → 1-3s back-off → attempt-2
    # success, or two independent failures that both recover on the second
    # retry.
    agent = SiliconFlowClient(config.models, model=config.models.agent_model, max_attempts=4, timeout_seconds=120)
    target = SiliconFlowClient(config.models, model=config.models.target_model, max_attempts=4, timeout_seconds=120)
    judge = SiliconFlowClient(config.models, model=config.models.agent_model, max_attempts=4, timeout_seconds=120)
    points = list(DOMAIN_POINTS[domain])
    # If the config file declares its own subtasks, prefer those so the
    # ablation runs can target a smaller knowledge-point subset without
    # editing DOMAIN_POINTS (which would change every other invocation).
    config_points = [kp for sub in config.subtasks for kp in sub.knowledge_points]
    if config_points:
        valid = [point for point in config_points if point in DOMAIN_POINTS[domain]]
        if valid:
            points = valid
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    checkpoint = run_dir / "samples.jsonl"
    events_path = run_dir / "events.jsonl"
    # Per-round diversity bookkeeping.  We keep two parallel counters:
    # - unique question gain counts candidate prompts whose novelty label is
    #   not DUPLICATE and that were not seen verbatim before.
    # - behavioral richness tracks failure clusters; the ablation reuses the
    #   "key_point" as the cluster label, which is the most faithful stand-in
    #   available without a signature extractor.
    history_prompts: List[str] = []
    history_token_sets: Dict[str, Tuple[str, ...]] = {}
    previous_cluster_count = 0
    current_round_labels: List[str] = []
    seen_clusters: set = set()
    for point_index, point in enumerate(points):
        seed_data = _json_call(agent, _seed_prompt(domain, point))
        questions = seed_data.get("questions", [])
        if not isinstance(questions, list) or len(questions) != 2:
            raise RuntimeError(f"invalid seed response for {point}")
        history: List[Dict[str, Any]] = []
        for ordinal, item in enumerate(questions):
            record = _evaluate(domain, point, "seed", 0, ordinal, item, agent, target, judge)
            history.append(record); rows.append(record)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            history_prompts.append(record["prompt"])
            history_token_sets[record["prompt"]] = _normalize_tokens(record["prompt"])
            current_round_labels.append(record["key_point"])
            seen_clusters.add(record["key_point"])
        _emit_diversity_event(
            events_path, domain, "seed", 0, rows, history_prompts,
            history_token_sets, current_round_labels, seen_clusters,
            previous_cluster_count,
            candidates_per_round=candidates_per_round,
            use_novelty_filter=use_novelty_filter,
        )
        current_round_labels.clear()
        for iteration in (1, 2):
            chosen = _generate_adaptive_question(
                domain, point, history, agent,
                candidates_per_round=candidates_per_round,
                use_novelty_filter=use_novelty_filter,
                novelty_top_k=novelty_top_k,
                history_prompts=history_prompts,
                history_token_sets=history_token_sets,
                events_path=events_path,
                phase="adaptive", iteration=iteration,
            )
            if chosen is None:
                # No candidates survived filtering and we already recorded the
                # failure in the events stream; emit a placeholder sample so
                # the per-iteration accounting stays aligned.
                record = {
                    "domain": domain, "point": point, "key_point": point,
                    "phase": "adaptive", "iteration": iteration, "ordinal": 0,
                    "prompt": "", "answer": "", "reference_answer": "",
                    "comparison": "", "score": 0.0, "is_weakness": True,
                    "status": "rejected",
                    "request_count": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0,
                }
            else:
                record = _evaluate(domain, point, "adaptive", iteration, 0, chosen, agent, target, judge)
                history_prompts.append(record["prompt"])
                history_token_sets[record["prompt"]] = _normalize_tokens(record["prompt"])
                current_round_labels.append(record["key_point"])
                seen_clusters.add(record["key_point"])
            history.append(record); rows.append(record)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        _emit_diversity_event(
            events_path, domain, "adaptive", 2, rows, history_prompts,
            history_token_sets, current_round_labels, seen_clusters,
            previous_cluster_count,
            candidates_per_round=candidates_per_round,
            use_novelty_filter=use_novelty_filter,
        )
        previous_cluster_count = len(seen_clusters)
        current_round_labels.clear()
    _write_summary(run_dir, domain, rows)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_type": "fixed_ablation_adapter",
                "formal_runner": False,
                "dynamic_assessor": False,
                "controller_enabled": False,
                "fixed_iterations": 2,
                "fixed_taxonomy": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "domain": domain,
        "samples": len(rows),
        "run_dir": str(run_dir),
        "candidates_per_round": candidates_per_round,
        "use_novelty_filter": use_novelty_filter,
    }


def _generate_adaptive_question(
    domain: str,
    point: str,
    history: List[Dict[str, Any]],
    agent: SiliconFlowClient,
    *,
    candidates_per_round: int,
    use_novelty_filter: bool,
    novelty_top_k: int,
    history_prompts: List[str],
    history_token_sets: Dict[str, Tuple[str, ...]],
    events_path: Path,
    phase: str,
    iteration: int,
) -> "Dict[str, Any] | None":
    """Pick the next adaptive question, optionally through the novelty filter."""
    if not use_novelty_filter or candidates_per_round <= 1:
        # Legacy path: one candidate, no filtering.  This is the "ablation"
        # branch that reproduces the original 2026-09-30 behaviour.
        payload = _json_call(agent, _adaptive_prompt(domain, point, history))
        return {"prompt": payload.get("prompt", ""), "key_point": payload.get("key_point", point)}

    count = max(1, candidates_per_round)
    try:
        payload = _json_call(
            agent, _adaptive_candidates_prompt(domain, point, history, count)
        )
    except RuntimeError:
        # The structured candidate prompt occasionally fails; fall back to
        # single-shot generation so the run still produces a sample.
        payload = _json_call(agent, _adaptive_prompt(domain, point, history))
        return {"prompt": payload.get("prompt", ""), "key_point": payload.get("key_point", point)}

    raw_questions = payload.get("questions", [])
    if not isinstance(raw_questions, list):
        raw_questions = []
    # Deduplicate candidates against history before asking the LLM to label
    # them: a verbatim repeat would always be rejected by the filter anyway,
    # and skipping it saves a judge call per duplicate.
    unique_candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        prompt_text = str(item.get("prompt", "")).strip()
        key_point = str(item.get("key_point", point)).strip()
        if not prompt_text or prompt_text in history_prompts or prompt_text in seen:
            continue
        seen.add(prompt_text)
        unique_candidates.append({"prompt": prompt_text, "key_point": key_point})

    if not unique_candidates:
        # All candidates were duplicates of history; fall back to a single
        # fresh generation so the round still has a question.
        payload = _json_call(agent, _adaptive_prompt(domain, point, history))
        return {"prompt": payload.get("prompt", ""), "key_point": payload.get("key_point", point)}

    # Score each candidate against the top-k nearest history neighbours
    # using Jaccard distance.  This is the same role that embeddings play
    # in the live runner, but without an external embedding server.
    scored: List[Dict[str, Any]] = []
    for item in unique_candidates:
        tokens = _normalize_tokens(item["prompt"])
        if not history_token_sets:
            nearest = 0.0
        else:
            sims = sorted(
                (_jaccard(tokens, history_token_sets[hp]) for hp in history_prompts),
                reverse=True,
            )[:3]
            nearest = sims[0] if sims else 0.0
        scored.append({**item, "tokens": tokens, "nn_similarity": nearest})

    # LLM judge labels each candidate as duplicate / related / distinct.
    decisions: List[Dict[str, Any]] = []
    for item in scored:
        try:
            judge = _call(
                agent, _novelty_judge_prompt(item["prompt"], history_prompts[-3:]),
                temperature=0.0,
            )
            label = _parse_novelty_label(judge.content)
        except Exception:
            label = "distinct"
        decisions.append({**item, "label": label})

    selected = [d for d in decisions if d["label"] != "duplicate"]
    if not selected:
        _emit_event(
            events_path, "candidate_filter",
            {
                "phase": phase, "iteration": iteration,
                "candidates": [
                    {"prompt": d["prompt"], "label": d["label"],
                     "semantic_novelty": 1.0 - d["nn_similarity"]}
                    for d in decisions
                ],
                "rejected_all": True,
            },
        )
        return None

    # Rank by novelty (lower nearest-neighbour similarity is more novel) and
    # then by recency of new content, taking up to novelty_top_k.
    selected.sort(key=lambda d: d["nn_similarity"])
    picked = selected[: max(1, novelty_top_k)]
    # Always take exactly one question for the round so the bookkeeping stays
    # comparable across ablations; novelty_top_k > 1 just gives a second
    # fallback if the top pick turns out to be unusable downstream.
    for pick in picked:
        if pick["prompt"]:
            _emit_event(
                events_path, "candidate_filter",
                {
                    "phase": phase, "iteration": iteration,
                    "candidates": [
                        {"prompt": d["prompt"], "label": d["label"],
                         "semantic_novelty": 1.0 - d["nn_similarity"]}
                        for d in decisions
                    ],
                    "selected_count": 1,
                },
            )
            return {"prompt": pick["prompt"], "key_point": pick["key_point"]}
    return None


def _emit_diversity_event(
    events_path: Path,
    domain: str,
    phase: str,
    iteration: int,
    rows: List[Dict[str, Any]],
    history_prompts: List[str],
    history_token_sets: Dict[str, Tuple[str, ...]],
    current_round_labels: List[str],
    seen_clusters: set,
    previous_cluster_count: int,
    *,
    candidates_per_round: int,
    use_novelty_filter: bool,
) -> None:
    """Compute semantic + behavioral metrics for the current snapshot and emit.

    The bookkeeping follows the same contract as :mod:`diversity`: a
    ``SemanticMetrics`` block summarises how novel the running question set is,
    and a ``BehavioralMetrics`` block reports new failure clusters.

    For the semantic side we deliberately use a Jaccard-distance signal instead
    of the 2-dim proxy in :mod:`diversity.semantic_metrics`, because that
    function expects dense embedding vectors.  The mean Jaccard distance from
    each question in the current round to the previous-round questions is a
    faithful stand-in for the cosine-based novelty score and produces numbers
    in the [0, 1] range that are directly comparable across ablations.
    """
    if not history_prompts:
        return
    # Split into "previous round" (everything but the last batch) and "current
    # round" (the last batch of valid samples, up to 2 adaptive candidates).
    # We rely on the fact that the seed batch and each adaptive round append
    # exactly len(current_round_labels) prompts.
    current_size = max(1, len(current_round_labels))
    current_prompts = history_prompts[-current_size:] if current_size else []
    previous_prompts = history_prompts[:-current_size] if current_size else history_prompts
    if previous_prompts and current_prompts:
        novelties: List[float] = []
        for prompt in current_prompts:
            tokens = history_token_sets.get(prompt) or _normalize_tokens(prompt)
            sims = sorted(
                (_jaccard(tokens, history_token_sets.get(prev) or _normalize_tokens(prev))
                 for prev in previous_prompts),
                reverse=True,
            )[:3]
            nearest = sims[0] if sims else 0.0
            novelties.append(1.0 - nearest)
        mean_novelty = sum(novelties) / len(novelties) if novelties else 0.0
        unique_gain = sum(1 for prompt in current_prompts if prompt not in previous_prompts)
    else:
        mean_novelty = 1.0
        unique_gain = len(current_prompts)
    target_calls = max(1, len(history_prompts))
    beh = behavioral_metrics(
        previous_count=previous_cluster_count,
        current_labels=current_round_labels,
        target_calls=target_calls,
    )
    _emit_event(
        events_path, "diversity_metrics",
        {
            "domain": domain, "phase": phase, "iteration": iteration,
            "samples_so_far": len(rows),
            "unique_question_gain": unique_gain,
            "mean_semantic_novelty": round(mean_novelty, 4),
            "behavioral_richness": beh.richness,
            "behavioral_gain": beh.gain,
            "discovery_rate": beh.discovery_rate,
            "target_calls": target_calls,
            "candidates_per_round": candidates_per_round,
            "use_novelty_filter": use_novelty_filter,
            "seen_clusters": sorted(seen_clusters),
        },
    )


def _evaluate(domain, point, phase, iteration, ordinal, item, agent, target, judge):
    prompt = str(item.get("prompt", "")).strip()
    key_point = str(item.get("key_point", point)).strip()
    target_result = _call(target, prompt, temperature=0.0)
    reference_result = _call(agent, _reference_prompt(domain, prompt), temperature=0.0)
    judge_prompt = _judge_prompt(domain, prompt, target_result.content, reference_result.content)
    judge_result = _call(judge, judge_prompt, temperature=0.0)
    try:
        score = parse_score(judge_result.content)
    except Exception:
        # DeepSeek occasionally omits the bracketed score. Spend one bounded
        # repair call rather than aborting an otherwise valid long run.
        repair = _call(
            judge,
            judge_prompt + "\nReturn only the final line in exactly this form: Rating: [[7]].",
            temperature=0.0,
        )
        judge_result = repair
        score = parse_score(judge_result.content)
    usage = [target_result, reference_result, judge_result]
    return {
        "domain": domain, "point": point, "key_point": key_point,
        "phase": phase, "iteration": iteration, "ordinal": ordinal,
        "prompt": prompt, "answer": target_result.content,
        "reference_answer": reference_result.content, "comparison": judge_result.content,
        "score": score, "is_weakness": score < 4, "status": "valid",
        "request_count": sum(x.request_count for x in usage),
        "prompt_tokens": sum(x.usage.prompt_tokens for x in usage),
        "completion_tokens": sum(x.usage.completion_tokens for x in usage),
        "total_tokens": sum(x.usage.total_tokens for x in usage),
    }


def _write_summary(run_dir: Path, domain: str, rows: Sequence[Dict[str, Any]]) -> None:
    grouped = []
    for phase, iteration in (("seed", 0), ("adaptive", 1), ("adaptive", 2)):
        current = [r for r in rows if r["phase"] == phase and r["iteration"] == iteration]
        grouped.append({"phase": phase, "iteration": iteration, "total": len(current), "weak": sum(r["is_weakness"] for r in current), "isr": sum(r["is_weakness"] for r in current) / len(current) if current else 0.0, "mean_score": sum(r["score"] for r in current) / len(current) if current else 0.0})
    with (run_dir / "iterations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("phase", "iteration", "total", "weak", "isr", "mean_score"))
        writer.writeheader(); writer.writerows(grouped)

    # Diversity trajectory across rounds.  Re-read the events stream because
    # the rows table only contains per-sample records, not per-round rollups.
    events_path = run_dir / "events.jsonl"
    diversity_rows: List[Dict[str, Any]] = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "diversity_metrics":
                continue
            diversity_rows.append({
                "phase": event.get("phase", ""),
                "iteration": event.get("iteration", 0),
                "samples_so_far": event.get("samples_so_far", 0),
                "unique_question_gain": event.get("unique_question_gain", 0),
                "mean_semantic_novelty": round(event.get("mean_semantic_novelty", 0.0), 4),
                "behavioral_richness": event.get("behavioral_richness", 0),
                "behavioral_gain": event.get("behavioral_gain", 0),
                "discovery_rate": round(event.get("discovery_rate", 0.0), 4),
            })
    if diversity_rows:
        with (run_dir / "diversity.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "phase", "iteration", "samples_so_far",
                    "unique_question_gain", "mean_semantic_novelty",
                    "behavioral_richness", "behavioral_gain", "discovery_rate",
                ),
            )
            writer.writeheader()
            writer.writerows(diversity_rows)

    total = len(rows); weak = sum(r["is_weakness"] for r in rows)
    report = [f"# {domain.title()} Small-Scale AutoDetect Run", "", "> **Fixed ablation adapter only. Not a formal AutoDetect run.**", "> Formal experiments must use `python -m autodetect_repro.cli run --config ...`.", "", f"- Samples: {total}", f"- ISR (score < 4): {weak / total:.2%}", f"- Mean score: {sum(r['score'] for r in rows) / total:.3f}", "", "| Phase | Iteration | Samples | ISR | Mean score |", "|---|---:|---:|---:|---:|"]
    report.extend(f"| {r['phase']} | {r['iteration']} | {r['total']} | {r['isr']:.2%} | {r['mean_score']:.3f} |" for r in grouped)
    if diversity_rows:
        report.extend([
            "", "## Question Diversity",
            "",
            "| Phase | Iteration | Samples | Unique gain | Mean novelty | Behavioral richness | Discovery rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        report.extend(
            "| {phase} | {iteration} | {samples_so_far} | {unique_question_gain} | "
            "{mean_semantic_novelty:.3f} | {behavioral_richness} | {discovery_rate:.2%} |".format(**row)
            for row in diversity_rows
        )
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(DOMAIN_POINTS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--candidates-per-round", type=int, default=1,
        help="Number of adaptive candidates generated per round; >1 enables "
        "the novelty filter path.",
    )
    parser.add_argument(
        "--no-novelty-filter", action="store_true",
        help="Disable the LLM-labelled novelty filter even when multiple "
        "candidates are generated (ablation switch).",
    )
    parser.add_argument(
        "--novelty-top-k", type=int, default=1,
        help="How many distinct candidates to keep after filtering.",
    )
    args = parser.parse_args()
    effective_filter = bool(
        not args.no_novelty_filter and args.candidates_per_round > 1
    )
    print(json.dumps(
        run(
            args.domain, args.config, args.run_dir,
            candidates_per_round=args.candidates_per_round,
            use_novelty_filter=effective_filter,
            novelty_top_k=args.novelty_top_k,
        ),
        ensure_ascii=False, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
