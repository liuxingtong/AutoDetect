import pytest

from autodetect_repro.parsing import ParseError, parse_json_object, parse_score


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"key_point": "lists", "prompt": "Write a list"}', {"key_point": "lists", "prompt": "Write a list"}),
        ('```json\n{"key_point": "tables", "prompt": "Write a table"}\n```', {"key_point": "tables", "prompt": "Write a table"}),
        ('<think>{"draft": true}</think>\n{"key_point": "json", "prompt": "Return JSON"}', {"key_point": "json", "prompt": "Return JSON"}),
        ('{"draft": true}</think>{"key_point": "markdown", "prompt": "Use headings"}', {"key_point": "markdown", "prompt": "Use headings"}),
        ('prefix {"nested": {"value": 1}, "text": "brace } in string"} suffix', {"nested": {"value": 1}, "text": "brace } in string"}),
    ],
)
def test_parse_json_object_returns_last_complete_object(text, expected):
    assert parse_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "reasoning only", "```json\nnot-json\n```", "{\"broken\":"])
def test_parse_json_object_rejects_missing_valid_content(text):
    with pytest.raises(ParseError, match="valid JSON object"):
        parse_json_object(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[[3]]", 3.0),
        ("Explanation. Rating: [[3.5]]", 3.5),
        ("draft [[8]]</think>final Rating: [[2]]", 2.0),
    ],
)
def test_parse_score_returns_last_finite_in_range_rating(text, expected):
    assert parse_score(text) == expected


@pytest.mark.parametrize(
    "text",
    ["No rating", "Rating: [[NaN]]", "Rating: [[0]]", "Rating: [[10.1]]"],
)
def test_parse_score_rejects_absent_non_finite_or_out_of_range_values(text):
    with pytest.raises(ParseError, match="score"):
        parse_score(text)
