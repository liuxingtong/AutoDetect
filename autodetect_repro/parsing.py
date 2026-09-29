"""Defensive parsers for model-produced JSON and judge scores."""

import json
import math
import re
from typing import Any, Dict, List


class ParseError(ValueError):
    """Raised when a model response cannot be parsed safely."""


def _without_balanced_reasoning(text: str) -> str:
    if text.count("<think>") != text.count("</think>"):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _object_candidates(text: str) -> List[str]:
    candidates = []
    start = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"' and depth:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def parse_json_object(text: str) -> Dict[str, Any]:
    """Return the last complete JSON object in a model response."""

    valid = []
    for candidate in _object_candidates(_without_balanced_reasoning(text)):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            valid.append(value)
    if not valid:
        raise ParseError("response does not contain a valid JSON object")
    return valid[-1]


_SCORE_PATTERN = re.compile(
    r"\[\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan|inf))\s*\]\]",
    flags=re.IGNORECASE,
)


def parse_score(text: str) -> float:
    """Return the last bracketed judge score in the inclusive range 1–10."""

    matches = _SCORE_PATTERN.findall(_without_balanced_reasoning(text))
    if not matches:
        raise ParseError("response does not contain a bracketed score")
    score = float(matches[-1])
    if not math.isfinite(score) or not 1.0 <= score <= 10.0:
        raise ParseError("score must be finite and between 1 and 10")
    return score
