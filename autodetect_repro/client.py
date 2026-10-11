"""Bounded OpenAI-compatible chat client for SiliconFlow."""

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import requests

from .types import ModelConfig


class ApiError(RuntimeError):
    """A redacted provider or transport failure."""


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: TokenUsage
    request_count: int


class SiliconFlowClient:
    """Call one configured model with bounded retry behavior."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        model: str,
        max_attempts: int,
        timeout_seconds: int,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = config.base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": "Bearer {}".format(config.api_key),
            "Content-Type": "application/json",
        }
        self.model = model
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._sleep = sleep

    def _error(self, attempt: int, status: Optional[int], reason: str) -> ApiError:
        status_text = "transport" if status is None else "status {}".format(status)
        return ApiError(
            "SiliconFlow {} failure for model {} on request {}: {}".format(
                status_text, self.model, attempt, reason
            )
        )

    @staticmethod
    def _chat_result(payload: Mapping[str, Any], attempt: int) -> ChatResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("missing assistant content")

        raw_usage = payload.get("usage")
        usage: Dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(
            usage.get("total_tokens", prompt_tokens + completion_tokens)
        )
        return ChatResult(
            content=content,
            usage=TokenUsage(prompt_tokens, completion_tokens, total_tokens),
            request_count=attempt,
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(1, self.max_attempts + 1):
            status = None
            retryable = True
            reason = "request failed"
            try:
                response = self._session.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                status = response.status_code
                if status == 200:
                    try:
                        body = response.json()
                        if not isinstance(body, dict):
                            raise ValueError("response body is not an object")
                        return self._chat_result(body, attempt)
                    except (TypeError, ValueError, KeyError):
                        reason = "invalid success response"
                else:
                    retryable = status == 429 or status >= 500
                    reason = "provider rejected request"
            except (requests.Timeout, requests.ConnectionError):
                reason = "transient transport error"
            except requests.RequestException:
                # Treat all RequestException subclasses as retryable so that
                # one-off DNS hiccups, reset packets, and other low-level network
                # glitches never abort a sample.
                reason = "non-retryable transport error"
                retryable = True

            if not retryable or attempt == self.max_attempts:
                raise self._error(attempt, status, reason)
            # Jittered exponential back-off: uniform random fraction of the
            # nominal delay so that many concurrent batches do not re-stampede
            # the API at exactly the same moment.
            base_delay = 2 ** (attempt - 1)
            jitter = random.random() * base_delay
            self._sleep(float(base_delay + jitter))

        raise self._error(self.max_attempts, None, "attempt limit reached")
