from pathlib import Path

import pytest

from autodetect_repro.client import ChatResult, TokenUsage
from autodetect_repro.config import load_config
from autodetect_repro.roles import Assessor, Evaluator, Examiner, Questioner, RoleError
from autodetect_repro.types import KnowledgePoint, Question, SampleRecord


class FakeClient:
    def __init__(self, name, responses, log=None):
        self.name = name
        self.responses = list(responses)
        self.calls = []
        self.log = log if log is not None else []

    def complete(self, messages, *, temperature, max_tokens):
        self.calls.append((messages, temperature, max_tokens))
        self.log.append(self.name)
        content = self.responses.pop(0)
        return ChatResult(content, TokenUsage(10, 5, 15), 1)


@pytest.fixture
def config():
    return load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )


def point(name="list format"):
    return KnowledgePoint("format", "text format", name, False)


def question(prompt="Return a three-item JSON list", phase="seed", iteration=0):
    return Question(
        group="format",
        subtask="text format",
        knowledge_point="list format",
        key_point="fixed list structure",
        prompt=prompt,
        phase=phase,
        iteration=iteration,
        ordinal=0,
    )


def record(score, prompt_text):
    return SampleRecord(
        sample_id="sample-{}".format(score),
        group="format",
        subtask="text format",
        knowledge_point="list format",
        phase="seed",
        iteration=0,
        ordinal=0,
        prompt=prompt_text,
        key_point="list constraint",
        answer="answer",
        reference_answer="reference",
        comparison="Rating: [[{}]]".format(score),
        score=score,
        is_weakness=score < 4,
        status="valid",
        error=None,
        request_count=3,
        prompt_tokens=30,
        completion_tokens=15,
        total_tokens=45,
    )


def test_examiner_preserves_exact_configured_scope_and_filters_extra(config):
    response = {
        "selected_points": [
            {
                "group": subtask.group,
                "subtask": subtask.name,
                "knowledge_point": knowledge_point,
            }
            for subtask in config.subtasks
            for knowledge_point in subtask.knowledge_points
        ]
        + [
            {
                "group": "unconfigured",
                "subtask": "surprise",
                "knowledge_point": "should not appear",
            }
        ]
    }
    import json

    examiner = Examiner(config, FakeClient("agent", [json.dumps(response)]))

    selected = examiner.select_taxonomy()

    assert len(selected) == 12
    assert [(item.group, item.subtask, item.name) for item in selected] == [
        (subtask.group, subtask.name, name)
        for subtask in config.subtasks
        for name in subtask.knowledge_points
    ]


def test_questioner_generates_exactly_two_distinct_seed_questions(config):
    client = FakeClient(
        "agent",
        [
            '{"questions": ['
            '{"key_point": "list count", "prompt": "Return exactly three bullets"},'
            '{"key_point": "list nesting", "prompt": "Return a two-level list"}'
            "]}"
        ],
    )
    questioner = Questioner(config, client)

    questions = questioner.generate_seeds(point())

    assert len(questions) == 2
    assert len({item.prompt.casefold() for item in questions}) == 2
    assert [item.ordinal for item in questions] == [0, 1]
    assert all(item.phase == "seed" and item.iteration == 0 for item in questions)


def test_questioner_repairs_malformed_structured_output_once(config):
    client = FakeClient(
        "agent",
        [
            "not json",
            '{"questions": ['
            '{"key_point": "one", "prompt": "First prompt"},'
            '{"key_point": "two", "prompt": "Second prompt"}'
            "]}",
        ],
    )

    questions = Questioner(config, client).generate_seeds(point())

    assert len(questions) == 2
    assert len(client.calls) == 2
    repair_request = client.calls[1][0][0]["content"]
    assert "exactly 2" in repair_request
    assert "not json" in repair_request


def test_questioner_rejects_duplicate_seed_prompts_after_one_repair(config):
    duplicate = (
        '{"questions": ['
        '{"key_point": "one", "prompt": "Same prompt"},'
        '{"key_point": "two", "prompt": " same   prompt "}'
        "]}"
    )
    client = FakeClient("agent", [duplicate, duplicate])

    with pytest.raises(RoleError, match="distinct seed questions"):
        Questioner(config, client).generate_seeds(point())

    assert len(client.calls) == 2


def test_adaptive_question_receives_low_score_first_compact_history(config):
    client = FakeClient(
        "agent",
        ['{"key_point": "compound list", "prompt": "Return nested numbered lists"}'],
    )
    history = [record(8.0, "easy prompt"), record(2.0, "hard prompt")]

    generated = Questioner(config, client).generate_adaptive(
        point(), history, iteration=3
    )

    sent_prompt = client.calls[0][0][0]["content"]
    assert sent_prompt.index("hard prompt") < sent_prompt.index("easy prompt")
    assert '"iteration": 3' in sent_prompt
    assert generated.phase == "adaptive"
    assert generated.iteration == 3


def test_evaluator_calls_target_reference_and_judge_in_order_and_marks_weakness():
    call_order = []
    target = FakeClient("target", ["model answer"], call_order)
    agent = FakeClient(
        "agent",
        ["reference answer", "Incorrect constraint handling. Rating: [[3]]"],
        call_order,
    )
    evaluator = Evaluator(target, agent, max_tokens=256)

    result = evaluator.evaluate(question())

    assert call_order == ["target", "agent", "agent"]
    assert result.answer == "model answer"
    assert result.reference_answer == "reference answer"
    assert result.score == 3.0
    assert result.is_weakness is True
    assert result.request_count == 3
    assert result.total_tokens == 45
    judge_prompt = agent.calls[1][0][0]["content"]
    assert "incorrect" in judge_prompt.casefold()
    assert "above 3" in judge_prompt.casefold()


@pytest.mark.parametrize(
    ("response", "accepted"),
    [
        ('{"task_name": "compound formatting", "explanation": "Failures combine constraints"}', True),
        ('{"task_name": "list format", "explanation": "Already present"}', False),
        ('{"task_name": "", "explanation": "Empty"}', False),
        ('{"stop": true, "explanation": "Taxonomy is sufficient"}', False),
    ],
)
def test_assessor_accepts_only_one_distinct_non_empty_point(
    config, response, accepted
):
    subtask = config.subtasks[0]
    assessor = Assessor(config, FakeClient("agent", [response]))

    decision = assessor.propose(subtask, [record(2.0, "hard prompt")])

    assert decision.accepted is accepted
    if accepted:
        assert decision.point is not None
        assert decision.point.name == "compound formatting"
        assert decision.point.discovered is True
    else:
        assert decision.point is None
