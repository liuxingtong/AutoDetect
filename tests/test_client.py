from typing import Any, Dict, List

import pytest
import requests

from autodetect_repro.client import ApiError, SiliconFlowClient
from autodetect_repro.types import ModelConfig


class FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class ScriptedSession:
    def __init__(self, events: List[Any]):
        self.events = list(events)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def model_config(secret="unit-test-secret"):
    return ModelConfig(
        base_url="https://api.siliconflow.cn/v1",
        agent_model="deepseek-ai/DeepSeek-V4-Flash",
        target_model="Qwen/Qwen2.5-7B-Instruct",
        api_key_env="SILICONFLOW_API_KEY",
        api_key=secret,
    )


def success_response(content="answer"):
    return FakeResponse(
        200,
        {
            "id": "chat-1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        },
    )


def make_client(session, sleeps=None, max_attempts=4):
    return SiliconFlowClient(
        model_config(),
        model="deepseek-ai/DeepSeek-V4-Flash",
        max_attempts=max_attempts,
        timeout_seconds=120,
        session=session,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_complete_sends_expected_request_and_returns_usage():
    session = ScriptedSession([success_response()])
    client = make_client(session)

    result = client.complete(
        [{"role": "user", "content": "hello"}],
        temperature=0.25,
        max_tokens=321,
    )

    assert result.content == "answer"
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 18
    assert result.request_count == 1
    assert session.requests == [
        (
            "https://api.siliconflow.cn/v1/chat/completions",
            {
                "headers": {
                    "Authorization": "Bearer unit-test-secret",
                    "Content-Type": "application/json",
                },
                "json": {
                    "model": "deepseek-ai/DeepSeek-V4-Flash",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.25,
                    "max_tokens": 321,
                },
                "timeout": 120,
            },
        )
    ]


@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(429, {"message": "rate limited"}),
        FakeResponse(503, {"message": "overloaded"}),
        requests.Timeout("slow provider"),
    ],
)
def test_transient_failure_retries_with_bounded_backoff(failure):
    sleeps = []
    session = ScriptedSession([failure, failure, failure, success_response()])
    client = make_client(session, sleeps=sleeps)

    result = client.complete(
        [{"role": "user", "content": "hello"}], temperature=0, max_tokens=8
    )

    assert result.request_count == 4
    assert len(session.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_permanent_auth_or_model_failure_does_not_retry(status_code):
    sleeps = []
    session = ScriptedSession(
        [FakeResponse(status_code, {"message": "unit-test-secret"})]
    )
    client = make_client(session, sleeps=sleeps)

    with pytest.raises(ApiError) as caught:
        client.complete(
            [{"role": "user", "content": "unit-test-secret"}],
            temperature=0,
            max_tokens=8,
        )

    message = str(caught.value)
    assert "deepseek-ai/DeepSeek-V4-Flash" in message
    assert "request 1" in message
    assert "status 401" in message or "status {}".format(status_code) in message
    assert "unit-test-secret" not in message
    assert len(session.requests) == 1
    assert sleeps == []


def test_http_200_without_usable_choice_fails_after_exact_attempt_limit():
    sleeps = []
    invalid = FakeResponse(
        200,
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 0}},
    )
    session = ScriptedSession([invalid, invalid, invalid, invalid])
    client = make_client(session, sleeps=sleeps)

    with pytest.raises(ApiError, match="request 4"):
        client.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=8,
        )

    assert len(session.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_timeout_failure_stops_after_exact_attempt_limit():
    sleeps = []
    session = ScriptedSession([requests.Timeout("x")] * 4)
    client = make_client(session, sleeps=sleeps)

    with pytest.raises(ApiError, match="request 4"):
        client.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=8,
        )

    assert len(session.requests) == 4
    assert sleeps == [1.0, 2.0, 4.0]
