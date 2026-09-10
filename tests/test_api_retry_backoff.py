import pytest
import requests

from chcons.api_agent import OpenRouterReActAgent


class _StubResponse:
    def __init__(self, status_code: int, *, content: str = "", headers=None):
        self.status_code = status_code
        self.text = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")

    def json(self) -> dict:
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": self.text}}],
            "usage": {"completion_tokens": 1},
        }


def _agent() -> OpenRouterReActAgent:
    agent = object.__new__(OpenRouterReActAgent)
    agent.api_model = "test/model"
    agent.request_timeout = 1
    agent._headers = {}
    agent.api_incidents = []
    agent.api_retries = []
    agent.api_reasoning = []
    return agent


def _stub_posts(monkeypatch, responses):
    calls = []
    response_iter = iter(responses)

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return next(response_iter)

    monkeypatch.setattr("chcons.api_agent.requests.post", post)
    return calls


def test_429s_then_success_use_exponential_backoff(monkeypatch) -> None:
    calls = _stub_posts(
        monkeypatch,
        [
            _StubResponse(429, content="rate limited"),
            _StubResponse(429, content="rate limited"),
            _StubResponse(429, content="rate limited"),
            _StubResponse(200, content="recovered"),
        ],
    )
    sleeps = []
    monkeypatch.setattr("chcons.api_agent.time.sleep", sleeps.append)

    assert _agent()._chat([{"role": "user", "content": "question"}], 64) == "recovered"
    assert len(calls) == 4
    assert sleeps == [5, 10, 20]


def test_retry_after_is_honoured_and_capped(monkeypatch) -> None:
    _stub_posts(
        monkeypatch,
        [
            _StubResponse(429, content="rate limited", headers={"Retry-After": "45"}),
            _StubResponse(429, content="rate limited", headers={"Retry-After": "999"}),
            _StubResponse(200, content="recovered"),
        ],
    )
    sleeps = []
    monkeypatch.setattr("chcons.api_agent.time.sleep", sleeps.append)

    assert _agent()._chat([{"role": "user", "content": "question"}], 64) == "recovered"
    assert sleeps == [45, 120]


def test_eight_429s_raise_with_existing_message_format(monkeypatch) -> None:
    calls = _stub_posts(
        monkeypatch,
        [_StubResponse(429, content="temporarily rate-limited upstream") for _ in range(8)],
    )
    sleeps = []
    monkeypatch.setattr("chcons.api_agent.time.sleep", sleeps.append)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^OpenRouter call failed after 8 tries: "
            r"429: temporarily rate-limited upstream$"
        ),
    ):
        _agent()._chat([{"role": "user", "content": "question"}], 64)

    assert len(calls) == 8
    assert sleeps == [5, 10, 20, 40, 60, 60, 60, 60]


def test_400_is_not_retried(monkeypatch) -> None:
    calls = _stub_posts(monkeypatch, [_StubResponse(400, content="bad request")])
    sleeps = []
    monkeypatch.setattr("chcons.api_agent.time.sleep", sleeps.append)

    with pytest.raises(requests.HTTPError, match=r"^400 Client Error$"):
        _agent()._chat([{"role": "user", "content": "question"}], 64)

    assert len(calls) == 1
    assert sleeps == []
