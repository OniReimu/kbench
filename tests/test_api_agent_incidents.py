from chcons.api_agent import OpenRouterReActAgent


class _StubResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _agent() -> OpenRouterReActAgent:
    agent = object.__new__(OpenRouterReActAgent)
    agent.api_model = "test/model"
    agent.request_timeout = 1
    agent.max_retries = 1
    agent._headers = {}
    agent.api_incidents = []
    agent.api_retries = []
    agent.api_reasoning = []
    agent.max_new_tokens = 64
    return agent


def test_chat_records_only_empty_completion_incidents(monkeypatch) -> None:
    empty_payload = {
        "choices": [
            {
                "finish_reason": "length",
                "native_finish_reason": "max_output_tokens",
                "message": {"content": None, "refusal": None, "reasoning": None},
            }
        ],
        "usage": {
            "completion_tokens": 64,
            "completion_tokens_details": {"reasoning_tokens": 61},
        },
    }
    monkeypatch.setattr(
        "chcons.api_agent.requests.post",
        lambda *args, **kwargs: _StubResponse(empty_payload),
    )
    agent = _agent()

    assert agent._chat([{"role": "user", "content": "question"}], 64) == ""
    assert len(agent.api_incidents) == 1
    assert agent.api_incidents[0]["call_site"] == "step"
    assert agent.api_incidents[0]["content_was_null"] is True
    assert agent.api_incidents[0]["reasoning_tokens"] == 61

    normal_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "message": {"content": "Thought: healthy", "refusal": None},
            }
        ],
        "usage": {
            "completion_tokens": 4,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
    monkeypatch.setattr(
        "chcons.api_agent.requests.post",
        lambda *args, **kwargs: _StubResponse(normal_payload),
    )
    agent = _agent()

    assert agent._chat([{"role": "user", "content": "question"}], 64) == "Thought: healthy"
    assert agent.api_incidents == []


def test_summary_and_step_incidents_record_their_call_sites(monkeypatch) -> None:
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": None}}],
        "usage": {"completion_tokens": 12},
    }
    monkeypatch.setattr(
        "chcons.api_agent.requests.post",
        lambda *args, **kwargs: _StubResponse(payload),
    )
    agent = _agent()

    assert agent.elicit_summary("Alice") == ""
    assert [incident["call_site"] for incident in agent.api_incidents] == ["summary"]

    agent.api_incidents = []
    assert agent._generate_block("user\x1fquestion\x1d") == ""
    assert [incident["call_site"] for incident in agent.api_incidents] == ["step"]


def test_reasoning_effort_is_added_only_when_requested(monkeypatch) -> None:
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": "healthy"}}],
        "usage": {"completion_tokens": 1},
    }
    posted_bodies = []

    def post(*args, **kwargs):
        posted_bodies.append(kwargs["json"])
        return _StubResponse(payload)

    monkeypatch.setattr("chcons.api_agent.requests.post", post)
    agent = _agent()

    agent.reasoning_effort = "high"
    assert agent._chat([{"role": "user", "content": "question"}], 64) == "healthy"
    assert posted_bodies[-1]["reasoning"] == {"effort": "high"}

    agent.reasoning_effort = None
    assert agent._chat([{"role": "user", "content": "question"}], 64) == "healthy"
    assert "reasoning" not in posted_bodies[-1]


def _payload(
    content,
    *,
    finish_reason="stop",
    completion_tokens=12,
    reasoning_tokens=None,
    reasoning=None,
    reasoning_details=None,
):
    details = {}
    if reasoning_tokens is not None:
        details["reasoning_tokens"] = reasoning_tokens
    message = {"content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    if reasoning_details is not None:
        message["reasoning_details"] = reasoning_details
    return {
        "choices": [{"finish_reason": finish_reason, "message": message}],
        "usage": {
            "completion_tokens": completion_tokens,
            "completion_tokens_details": details,
        },
    }


def _stub_posts(monkeypatch, payloads):
    posted_bodies = []
    responses = iter(payloads)

    def post(*args, **kwargs):
        posted_bodies.append(kwargs["json"])
        return _StubResponse(next(responses))

    monkeypatch.setattr("chcons.api_agent.requests.post", post)
    return posted_bodies


def test_stop_null_retries_once_and_recovers(monkeypatch) -> None:
    posted_bodies = _stub_posts(
        monkeypatch,
        [_payload(None, completion_tokens=143, reasoning_tokens=0), _payload("healthy")],
    )
    agent = _agent()

    assert agent._chat([{"role": "user", "content": "question"}], 64) == "healthy"
    assert len(posted_bodies) == 2
    assert posted_bodies[0] is posted_bodies[1]
    assert agent.api_incidents == []
    assert agent.api_retries == [{
        "call_site": "step",
        "finish_reason": "stop",
        "completion_tokens": 143,
        "reasoning_tokens": 0,
        "recovered": True,
    }]


def test_stop_null_retries_once_then_records_one_incident(monkeypatch) -> None:
    posted_bodies = _stub_posts(
        monkeypatch,
        [_payload(None, reasoning_tokens=0), _payload(None, reasoning_tokens=0)],
    )
    agent = _agent()

    assert agent._chat(
        [{"role": "user", "content": "question"}], 64, call_site="summary"
    ) == ""
    assert len(posted_bodies) == 2
    assert len(agent.api_incidents) == 1
    assert agent.api_incidents[0]["call_site"] == "summary"
    assert agent.api_incidents[0]["retried"] is True
    assert agent.api_retries == [{
        "call_site": "summary",
        "finish_reason": "stop",
        "completion_tokens": 12,
        "reasoning_tokens": 0,
        "recovered": False,
    }]


def test_length_null_does_not_retry(monkeypatch) -> None:
    posted_bodies = _stub_posts(monkeypatch, [_payload(None, finish_reason="length")])
    agent = _agent()

    assert agent._chat([{"role": "user", "content": "question"}], 64) == ""
    assert len(posted_bodies) == 1
    assert len(agent.api_incidents) == 1
    assert agent.api_retries == []


def test_content_and_reasoning_returns_content_and_records_reasoning(monkeypatch) -> None:
    posted_bodies = _stub_posts(
        monkeypatch, [_payload("Thought: visible", reasoning="private analysis")]
    )
    agent = _agent()

    assert agent._chat(
        [{"role": "user", "content": "question"}], 64, call_site="summary"
    ) == "Thought: visible"
    assert len(posted_bodies) == 1
    assert agent.api_reasoning == [
        {"call_site": "summary", "text": "private analysis"}
    ]


def test_reasoning_only_empty_is_recorded_and_not_retried(monkeypatch) -> None:
    posted_bodies = _stub_posts(
        monkeypatch,
        [_payload(None, reasoning_tokens=0, reasoning="Thought: hidden")],
    )
    agent = _agent()

    assert agent._chat([{"role": "user", "content": "question"}], 64) == ""
    assert len(posted_bodies) == 1
    assert len(agent.api_incidents) == 1
    assert agent.api_incidents[0]["reasoning_present"] is True
    assert agent.api_retries == []
    assert agent.api_reasoning == [
        {"call_site": "step", "text": "Thought: hidden"}
    ]


def test_reasoning_details_are_joined_when_reasoning_is_absent(monkeypatch) -> None:
    _stub_posts(
        monkeypatch,
        [_payload(
            "visible",
            reasoning_details=[{"text": "first "}, {"type": "reasoning.text", "text": "second"}],
        )],
    )
    agent = _agent()

    assert agent._chat([{"role": "user", "content": "question"}], 64) == "visible"
    assert agent.api_reasoning == [
        {"call_site": "step", "text": "first second"}
    ]


def test_healthy_first_completion_does_not_retry(monkeypatch) -> None:
    posted_bodies = _stub_posts(monkeypatch, [_payload("healthy")])
    agent = _agent()

    assert agent._chat([{"role": "user", "content": "question"}], 64) == "healthy"
    assert len(posted_bodies) == 1
    assert agent.api_incidents == []
    assert agent.api_retries == []
