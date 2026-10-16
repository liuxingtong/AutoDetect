"""Prompt contracts adapted from the released AutoDetect implementation."""

import json
from typing import Optional, Sequence

from .types import KnowledgePoint, Question, SampleRecord, SubtaskConfig


def controller_prompt(taxonomy, branch_statistics, recent_cases, budget: int) -> str:
    return (
        "Act as the AutoDetect Adaptive Controller. Choose exploit, explore, verify, or restart "
        "to maximize confirmed non-duplicate failures under a strict budget. Return JSON only. "
        "Do not modify taxonomy or exceed the supplied budget.\n"
        "taxonomy={taxonomy}\nbranch_statistics={stats}\nrecent_cases={cases}\nremaining_budget={budget}"
    ).format(taxonomy=taxonomy, stats=branch_statistics, cases=recent_cases, budget=budget)


def failure_signature_prompt(question: str, answer: str, reference: str, evidence: str) -> str:
    return (
        "Act as an AutoDetect failure analyzer. Describe only observable behavior, not an internal cause. "
        "Return JSON with violation, observed_behavior, trigger_context, and evidence. "
        "Use is_failure false when evidence does not establish a failure.\n"
        "question={question}\nanswer={answer}\nreference={reference}\nprogrammatic_evidence={evidence}"
    ).format(question=question, answer=answer, reference=reference, evidence=evidence)


def behavioral_clustering_prompt(signature, representatives) -> str:
    return (
        "Compare one Failure Signature with representative behavior categories. Return JSON only with "
        "label: Same or Different, category_name, and a reason. Different requires a distinct observed behavior "
        "or a materially different trigger condition; do not infer hidden causes. If evidence is "
        "ambiguous, choose Different so the pattern is preserved as a peer category. For Different, "
        "category_name must be a concise human-readable taxonomy name; for Same it may be empty.\n"
        "new_signature={signature}\nrepresentatives={representatives}"
    ).format(signature=signature, representatives=representatives)


def promotion_prompt(branch, evidence) -> str:
    return (
        "Act as the AutoDetect Examiner for taxonomy promotion. Return JSON only with decision: "
        "promote, continue_verification, or reject. Promote only when evidence includes multiple "
        "supporting cases, a repeat or controlled comparison, and an independently testable meaning.\n"
        "branch={branch}\nevidence={evidence}"
    ).format(branch=branch, evidence=evidence)


def examiner_prompt(subtasks: Sequence[SubtaskConfig]) -> str:
    taxonomy = [
        {
            "group": item.group,
            "subtask": item.name,
            "knowledge_points": list(item.knowledge_points),
        }
        for item in subtasks
    ]
    return (
        "Act as the AutoDetect Examiner. Normalize the configured "
        "instruction-following taxonomy without expanding its scope. Return only "
        "JSON with selected_points, an array of objects containing group, subtask, "
        "and knowledge_point. Include every configured knowledge point.\n"
        "Configured taxonomy:\n{}".format(
            json.dumps(taxonomy, ensure_ascii=False, sort_keys=True)
        )
    )


def seed_prompt(point: KnowledgePoint, count: int, taxonomy_context: str = "") -> str:
    taxonomy_block = (
        "\nHere is the full domain taxonomy:\n{taxonomy}\n\n"
    ).format(taxonomy=taxonomy_context) if taxonomy_context else "\n"
    return (
        "Act as the AutoDetect Questioner. Generate exactly {count} diverse, "
        "self-contained, solvable tests for the knowledge point "
        "{point!r} in subtask {subtask!r}. Express each constraint explicitly "
        "so it can be verified automatically.{taxonomy}"
        "Aim for prompts that would result in a low score (less than 4.0), "
        "indicating genuine difficulty or a subtle constraint the target model "
        "is likely to violate. Cover different angles (entity, scenario, or "
        "constraint variant). Do not repeat verbs or phrasing across entries. "
        "Return only JSON as {{\"questions\": [{{\"key_point\": \"...\", "
        "\"prompt\": \"...\"}}]}} with exactly {count} distinct entries."
    ).format(
        count=count,
        point=point.name,
        subtask=point.subtask,
        taxonomy=taxonomy_block,
    )


def adaptive_prompt(
    point: KnowledgePoint,
    history: Sequence[SampleRecord],
    iteration: int,
    *,
    strategy: Optional[str] = None,
    objective: Optional[str] = None,
) -> str:
    compact = [
        {
            "prompt": item.prompt,
            "key_point": item.key_point,
            "score": item.score,
        }
        for item in sorted(
            history,
            key=lambda item: (
                item.score is None,
                item.score if item.score is not None else 11.0,
            ),
        )[:5]
    ]
    context = {"iteration": iteration, "history": compact}
    strategy_clause = (
        "\nAdopt the search strategy {strategy!r}: {objective!r}."
    ).format(strategy=strategy, objective=objective) if strategy else ""
    return (
        "Act as the AutoDetect Questioner. Generate one new, solvable and distinct "
        "test for {point!r} that is likely to score below 4.0. "
        "Use lower-scoring history to expose a deeper weakness "
        "without copying a previous prompt. Each prompt should explore a different "
        "angle (boundary case, edge constraint, unusual combination, etc.). "
        "Return only JSON with key_point and prompt.\nSearch context:\n{context}{strategy_clause}"
    ).format(
        point=point.name,
        context=json.dumps(context, ensure_ascii=False, sort_keys=True),
        strategy_clause=strategy_clause,
    )


def adaptive_candidates_prompt(
    point: KnowledgePoint,
    history: Sequence[SampleRecord],
    iteration: int,
    count: int,
    *,
    strategy: Optional[str] = None,
    objective: Optional[str] = None,
) -> str:
    """Generate ``count`` distinct adaptive candidates for one objective.

    Used by the adaptive controller's ``candidate_budget``.  Each candidate
    must be self-contained and materially different from the others.
    """
    compact = [
        {
            "prompt": item.prompt,
            "key_point": item.key_point,
            "score": item.score,
        }
        for item in sorted(
            history,
            key=lambda item: (
                item.score is None,
                item.score if item.score is not None else 11.0,
            ),
        )[:5]
    ]
    context = {"iteration": iteration, "history": compact}
    strategy_clause = (
        "\nAdopt the search strategy {strategy!r}: {objective!r}."
    ).format(strategy=strategy, objective=objective) if strategy else ""
    return (
        "Act as the AutoDetect Questioner. Generate exactly {count} new, solvable "
        "and mutually distinct tests for the knowledge point {point!r}. Each "
        "candidate must be likely to score below 4.0. "
        "Each candidate must change a meaningful variable (entity, scenario, or "
        "constraint) and must not copy any prior prompt. Return only JSON as "
        "{{\"questions\": [{{\"key_point\": \"...\", \"prompt\": \"...\"}}]}} "
        "with exactly {count} entries.\nSearch context:\n{context}{strategy_clause}"
    ).format(
        count=count,
        point=point.name,
        context=json.dumps(context, ensure_ascii=False, sort_keys=True),
        strategy_clause=strategy_clause,
    )


def reference_prompt(question: Question) -> str:
    return (
        "Produce a high-quality reference response that follows every constraint "
        "in this instruction. Return only the response.\n\nInstruction:\n{}"
    ).format(question.prompt)


def judge_prompt(question: Question, answer: str, reference: str) -> str:
    return (
        "Act as an impartial judge. Compare the assistant answer with the reference "
        "and evaluate correctness, instruction following, relevance, and detail on "
        "a 1 to 10 scale. If the answer is incorrect or violates a required "
        "constraint, it must not score above 3. Explain briefly, then end with "
        "Rating: [[number]].\n\nQuestion:\n{question}\n\nReference answer:\n"
        "{reference}\n\nAssistant answer:\n{answer}"
    ).format(question=question.prompt, reference=reference, answer=answer)


def assessor_prompt(
    subtask: SubtaskConfig, history: Sequence[SampleRecord]
) -> str:
    bad_cases = [
        {
            "knowledge_point": item.knowledge_point,
            "prompt": item.prompt,
            "answer": item.answer,
            "score": item.score,
        }
        for item in sorted(
            (entry for entry in history if entry.score is not None),
            key=lambda entry: entry.score,
        )[:5]
    ]
    return (
        "Act as the AutoDetect Assessor for subtask {subtask!r}. Existing knowledge "
        "points are {points}. Infer at most one materially distinct missing point "
        "from the low-scoring cases. Return only JSON as {{\"task_name\": \"...\", "
        "\"explanation\": \"...\"}} or {{\"stop\": true, "
        "\"explanation\": \"...\"}}.\nLow-scoring cases:\n{cases}"
    ).format(
        subtask=subtask.name,
        points=json.dumps(list(subtask.knowledge_points), ensure_ascii=False),
        cases=json.dumps(bad_cases, ensure_ascii=False, sort_keys=True),
    )


def repair_prompt(original_instruction: str, response: str) -> str:
    return (
        "Repair the response into only the JSON object required by the original "
        "instruction. Preserve its intended content and add no prose.\n\nOriginal "
        "instruction:\n{instruction}\n\nResponse to repair:\n{response}"
    ).format(instruction=original_instruction[:4000], response=response[:4000])


def novelty_classify_prompt(
    candidate: str,
    neighbors: Sequence[str],
    knowledge_point: str = "",
) -> str:
    """Prompt the LLM to label a candidate question's novelty.

    Returns JSON with ``label: Duplicate | Related | Distinct | Uncertain``
    and a brief ``reason``.  Different requires a distinct observed behavior
    or a materially different trigger condition; do not infer hidden causes.
    """
    return (
        "Act as the AutoDetect Novelty Classifier for knowledge point {kp!r}. "
        "Compare the candidate question with the existing neighbors. Return "
        "only JSON as {{\"label\": \"Duplicate\"|\"Related\"|\"Distinct\"|"
        "\"Uncertain\", \"reason\": \"...\"}}.\n"
        "candidate={candidate}\nneighbors={neighbors}"
    ).format(
        kp=knowledge_point,
        candidate=candidate,
        neighbors=list(neighbors),
    )
