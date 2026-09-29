"""Prompt contracts adapted from the released AutoDetect implementation."""

import json
from typing import Sequence

from .types import KnowledgePoint, Question, SampleRecord, SubtaskConfig


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


def seed_prompt(point: KnowledgePoint, count: int) -> str:
    return (
        "Act as the AutoDetect Questioner. Generate exactly {count} diverse, "
        "self-contained instruction-following tests for the knowledge point "
        "{point!r} in subtask {subtask!r}. Express the constraint explicitly. "
        "Return only JSON as {{\"questions\": [{{\"key_point\": \"...\", "
        "\"prompt\": \"...\"}}]}} with exactly {count} distinct entries."
    ).format(count=count, point=point.name, subtask=point.subtask)


def adaptive_prompt(
    point: KnowledgePoint,
    history: Sequence[SampleRecord],
    iteration: int,
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
    return (
        "Act as the AutoDetect Questioner. Generate one new, solvable and distinct "
        "test for {point!r}. Use lower-scoring history to expose a deeper weakness "
        "without copying a previous prompt. Return only JSON with key_point and "
        "prompt.\nSearch context:\n{context}"
    ).format(
        point=point.name,
        context=json.dumps(context, ensure_ascii=False, sort_keys=True),
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
