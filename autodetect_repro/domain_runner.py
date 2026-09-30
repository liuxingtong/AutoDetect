"""Small, CPU-only math/code trend runs using the shared SiliconFlow client."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .client import SiliconFlowClient
from .config import load_config
from .parsing import parse_json_object, parse_score


DOMAIN_POINTS = {
    "math": (
        "algebra / arithmetic / basic_operations",
        "algebra / equations_and_inequalities / quadratic_equations",
        "geometry / trigonometry / trigonometric_identities",
        "analysis / derivatives / chain_rule",
    ),
    "code": (
        "string_processing / regular_expressions",
        "data_container_operations / filtering_and_selection",
        "logic_and_control_flow / exception_handling",
        "mathematics_and_algorithms / algorithm_design",
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


def run(domain: str, config_path: Path, run_dir: Path) -> Dict[str, Any]:
    env = __import__("os").environ
    config = load_config(config_path, env)
    agent = SiliconFlowClient(config.models, model=config.models.agent_model, max_attempts=4, timeout_seconds=120)
    target = SiliconFlowClient(config.models, model=config.models.target_model, max_attempts=4, timeout_seconds=120)
    points = list(DOMAIN_POINTS[domain])
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    checkpoint = run_dir / "samples.jsonl"
    for point_index, point in enumerate(points):
        seed_data = _json_call(agent, _seed_prompt(domain, point))
        questions = seed_data.get("questions", [])
        if not isinstance(questions, list) or len(questions) != 2:
            raise RuntimeError(f"invalid seed response for {point}")
        history: List[Dict[str, Any]] = []
        for ordinal, item in enumerate(questions):
            record = _evaluate(domain, point, "seed", 0, ordinal, item, agent, target)
            history.append(record); rows.append(record)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        for iteration in (1, 2):
            item = _json_call(agent, _adaptive_prompt(domain, point, history))
            record = _evaluate(domain, point, "adaptive", iteration, 0, item, agent, target)
            history.append(record); rows.append(record)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    _write_summary(run_dir, domain, rows)
    return {"domain": domain, "samples": len(rows), "run_dir": str(run_dir)}


def _evaluate(domain, point, phase, iteration, ordinal, item, agent, target):
    prompt = str(item.get("prompt", "")).strip()
    key_point = str(item.get("key_point", point)).strip()
    target_result = _call(target, prompt, temperature=0.0)
    reference_result = _call(agent, _reference_prompt(domain, prompt), temperature=0.0)
    judge_result = _call(agent, _judge_prompt(domain, prompt, target_result.content, reference_result.content), temperature=0.0)
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
        grouped.append({"phase": phase, "iteration": iteration, "total": len(current), "weak": sum(r["is_weakness"] for r in current), "isr": sum(r["is_weakness"] for r in current) / len(current), "mean_score": sum(r["score"] for r in current) / len(current)})
    with (run_dir / "iterations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("phase", "iteration", "total", "weak", "isr", "mean_score"))
        writer.writeheader(); writer.writerows(grouped)
    total = len(rows); weak = sum(r["is_weakness"] for r in rows)
    report = [f"# {domain.title()} Small-Scale AutoDetect Run", "", f"- Samples: {total}", f"- ISR (score < 4): {weak / total:.2%}", f"- Mean score: {sum(r['score'] for r in rows) / total:.3f}", "", "| Phase | Iteration | Samples | ISR | Mean score |", "|---|---:|---:|---:|---:|"]
    report.extend(f"| {r['phase']} | {r['iteration']} | {r['total']} | {r['isr']:.2%} | {r['mean_score']:.3f} |" for r in grouped)
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(DOMAIN_POINTS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.domain, args.config, args.run_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
