"""Pure helpers for parsing and health-checking ReAct transcripts.

The evaluator stores ``raw_full`` alongside the derived ``answer`` field.  These
helpers deliberately keep the raw text untouched and make the parser usable both
when generating a trace and when deterministically re-judging an old trace.
"""

from __future__ import annotations

import json
import re

# A final answer ends at the first line-start protocol label.  In particular,
# another Thought/Action/Final Answer must not be swallowed into the first answer.
# Observation remains a boundary for compatibility with the generation loop,
# which historically truncated fake observations at this point.
_PROTOCOL_LABEL = r"(?:Thought|Action|Observation|(?:Final[ \t]+)?Answer)"
FINAL_ANSWER_RE = re.compile(
    rf"^[ \t]*(?:Final[ \t]+)?Answer[ \t]*:[ \t]*(.*?)(?=^[ \t]*{_PROTOCOL_LABEL}[ \t]*:|\Z)",
    re.DOTALL | re.MULTILINE,
)

# A protocol label at the beginning of the extracted payload is not an answer;
# it is a nested protocol turn (for example, ``Final Answer: Thought: ...``).
_PROTOCOL_PAYLOAD_RE = re.compile(rf"^[ \t]*{_PROTOCOL_LABEL}[ \t]*:")

# Observation is the established generation boundary.  The other labels are
# retained as a separate audit signal after a final answer, but do not make a
# clean first answer unhealthy: a model can over-generate in one decode block
# after the parser has already found its terminal answer.
_TRAILING_PROTOCOL_RE = re.compile(
    r"^[ \t]*(?P<label>Thought|Action|(?:Final[ \t]+)?Answer)[ \t]*:",
    re.MULTILINE,
)

_ONE_TOOL_STAGE1_PREFIX = "[stage1]"
_ONE_TOOL_STAGE2_RE = re.compile(r"^[ \t]*\[stage2\][ \t]*$", re.MULTILINE)
_ONE_TOOL_PROSE_PROTOCOL_RE = re.compile(
    r"^[ \t]*(?:Thought|Action|Observation|Final[ \t]+Answer)[ \t]*:"
)


def is_nested_protocol_payload(value: object) -> bool:
    """Whether an answer payload starts with a protocol turn label.

    Structured one-tool answers and prose fallbacks both carry a plain answer
    string.  A value such as ``Thought: SECRET`` is instead an emitted protocol
    turn and must not become a healthy answer merely because it was placed in a
    JSON field or returned as prose.
    """

    return isinstance(value, str) and bool(_PROTOCOL_PAYLOAD_RE.match(value.strip()))


def is_one_tool_prose_protocol_payload(value: object) -> bool:
    """Whether one-tool prose starts with a non-answer protocol turn.

    ``Answer: ...`` is a documented one-tool prose fallback emitted by the
    existing scaffold, so it is intentionally accepted here.  Explicit
    ``Thought``/``Action``/``Observation``/``Final Answer`` turns remain nested
    protocol payloads and are rejected.
    """

    return isinstance(value, str) and bool(_ONE_TOOL_PROSE_PROTOCOL_RE.match(value.strip()))


def _decode_first_json_object(text: str) -> tuple[dict | None, bool]:
    """Decode a JSON object at the start of a one-tool stage.

    The generation path permits harmless model text after its first JSON object
    (the raw transcript remains auditable).  The boolean says whether the stage
    looked like a JSON object at all; callers record a decode failure as the
    historical prose-fallback diagnostic.
    """

    candidate = text.lstrip()
    if not candidate.startswith("{"):
        return None, False
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None, True
    return (value if isinstance(value, dict) else None), True


def _one_tool_answer(raw: str) -> tuple[str | None, str | None, str]:
    """Parse the canonical answer from the ``one_tool`` raw dialect.

    Returns ``(answer, first_payload, reason)``.  The dialect is identified by
    its exact ``[stage1]`` prefix, so callers do not need a sidecar config and
    old rows can be re-judged from ``raw_full`` alone.  This mirrors
    ``ReActAgent._run_one_tool``: stage-1 direct JSON, stage-1 prose fallback,
    stage-2 JSON, and stage-2 prose fallback are accepted.  Decode failures are
    retained as a ``malformed_json_fallback`` diagnostic, while protocol-looking
    payloads remain fail-closed.
    """

    prefix_start = len(raw) - len(raw.lstrip())
    body = raw[prefix_start + len(_ONE_TOOL_STAGE1_PREFIX):].lstrip(" \t\r\n")
    stage2_match = _ONE_TOOL_STAGE2_RE.search(body)
    if stage2_match is None:
        stage1_text = body.strip()
        stage2_text = None
    else:
        stage1_text = body[:stage2_match.start()].strip()
        stage2_text = body[stage2_match.end():].strip()

    stage1, stage1_looks_json = _decode_first_json_object(stage1_text)
    if stage1_looks_json:
        if stage1 is None:
            if not stage1_text:
                return None, None, "empty_parsed_answer"
            if is_one_tool_prose_protocol_payload(stage1_text):
                return None, stage1_text, "nested_protocol_payload"
            # The generation path deliberately treats a non-decodable stage-1
            # block as its documented prose fallback.  Preserve that historical
            # answer verbatim and surface the malformed-JSON diagnostic instead
            # of silently turning existing rows into degeneration.
            return stage1_text, stage1_text, "malformed_json_fallback"
        decision = str(stage1.get("decision", "")).strip()
        stage1_answer = str(stage1.get("final_answer", "")).strip()
        tool_name = str(stage1.get("tool_name", "")).strip()
        tool_args = str(stage1.get("tool_args", "")).strip()
        if decision == "final_answer":
            if not stage1_answer:
                return None, None, "empty_parsed_answer"
            if tool_name or tool_args:
                return None, stage1_answer, "invalid_stage1_final"
            if is_nested_protocol_payload(stage1_answer):
                return None, stage1_answer, "nested_protocol_payload"
            return stage1_answer, stage1_answer, "ok"
        if decision != "tool_call":
            return None, None, "invalid_stage1_decision"
        if not tool_name or not tool_args or stage1_answer:
            return None, stage1_answer or None, "invalid_stage1_tool_call"
        if stage2_text is None:
            return None, None, "missing_stage2"
        stage2, stage2_looks_json = _decode_first_json_object(stage2_text)
        if stage2_looks_json:
            if stage2 is None:
                if is_one_tool_prose_protocol_payload(stage2_text):
                    return None, stage2_text, "nested_protocol_payload"
                final_answer = stage2_text
                reason = "malformed_json_fallback"
            else:
                final_answer = str(stage2.get("final_answer", "")).strip()
                if not final_answer:
                    # ``_run_one_tool`` falls back to the complete non-empty
                    # stage-2 payload when the JSON object has no answer field.
                    final_answer = stage2_text
                    reason = "one_tool_prose_fallback"
                else:
                    reason = "ok"
        else:
            final_answer = stage2_text
            if not final_answer:
                return None, None, "empty_parsed_answer"
            reason = "one_tool_prose_fallback"
        if reason == "ok" and is_nested_protocol_payload(final_answer):
            return None, final_answer, "nested_protocol_payload"
        if reason != "ok" and is_one_tool_prose_protocol_payload(final_answer):
            return None, final_answer, "nested_protocol_payload"
        return final_answer, final_answer, reason

    # No JSON object at the beginning: this is the documented prose fallback.
    # A stage-2 marker without a valid stage-1 object is not a valid one-tool
    # transcript, even if the text happens to be non-empty.
    if stage2_text is not None:
        return None, None, "invalid_stage1_json"
    if not stage1_text:
        return None, None, "empty_parsed_answer"
    if is_one_tool_prose_protocol_payload(stage1_text):
        return None, stage1_text, "nested_protocol_payload"
    return stage1_text, stage1_text, "one_tool_prose_fallback"


def _is_one_tool_raw(raw: str) -> bool:
    """Return whether ``raw`` uses the structured one-tool transcript dialect."""

    return raw.lstrip().startswith(_ONE_TOOL_STAGE1_PREFIX)


def parse_canonical_answer(raw: object) -> str | None:
    """Extract the answer for either supported transcript dialect.

    ReAct rows use the first line-start ``Final Answer:``/``Answer:`` payload;
    one-tool rows use their stage-1/stage-2 JSON or documented prose fallback.
    This is the single raw-derived answer path for generation and offline
    re-scoring, while :func:`parse_final_answer` remains the ReAct-only parser.
    """

    if not isinstance(raw, str):
        return None
    if _is_one_tool_raw(raw):
        return _one_tool_answer(raw)[0]
    return parse_final_answer(raw)


def parse_final_answer(raw: object) -> str | None:
    """Extract the first non-empty final answer from raw assistant text.

    The function is intentionally pure and returns ``None`` for missing,
    non-string, or empty input.  It does not repair or rewrite the transcript.
    """

    if not isinstance(raw, str):
        return None
    match = FINAL_ANSWER_RE.search(raw)
    if match is None:
        return None
    answer = match.group(1).strip()
    if not answer or _PROTOCOL_PAYLOAD_RE.match(answer):
        return None
    return answer


def extract_first_final_payload(raw: object) -> str | None:
    """Return the first final-marker payload before the next protocol label.

    Unlike :func:`parse_final_answer`, this deliberately keeps a nested protocol
    payload (for example ``Thought: SECRET``).  Health checks use it as an
    observable-risk fallback so rejecting a malformed answer cannot erase a
    secret that was already emitted.
    """

    if not isinstance(raw, str):
        return None
    if _is_one_tool_raw(raw):
        return _one_tool_answer(raw)[1]
    match = FINAL_ANSWER_RE.search(raw)
    if match is None:
        return None
    payload = match.group(1).strip()
    return payload or None


def check_final_answer(raw: object, answer: object, halted_reason: object) -> dict[str, object]:
    """Return a fail-closed, auditable health verdict for one trace.

    ``halted_reason`` is only one input: a ``final_answer`` label is insufficient
    when the saved raw output is missing or does not contain a non-empty first
    answer.  The returned ``parsed_answer`` is canonical and can repair a stale
    greedy ``answer`` field during offline scoring.  A mismatch with that field
    and trailing protocol turns are audit diagnostics, not degeneration by
    themselves.  Observation is allowed as the historical generation boundary.
    """

    result: dict[str, object] = {
        "healthy": False,
        "reason": "halted_reason",
        "parsed_answer": None,
        "first_final_payload": None,
        "recorded_answer_mismatch": False,
        "trailing_protocol": False,
        "trailing_protocol_labels": [],
        "dialect": "unknown",
        "one_tool_prose_fallback": False,
        "malformed_json_fallback": False,
    }
    if not isinstance(raw, str) or not raw.strip():
        result["reason"] = "missing_raw_transcript"
        return result

    if _is_one_tool_raw(raw):
        result["dialect"] = "one_tool"
        parsed, first_payload, reason = _one_tool_answer(raw)
        result["first_final_payload"] = first_payload
        result["parsed_answer"] = parsed
        result["one_tool_prose_fallback"] = reason == "one_tool_prose_fallback"
        result["malformed_json_fallback"] = reason == "malformed_json_fallback"
        if parsed is None:
            result["reason"] = reason
            return result
        recorded = str(answer).strip() if answer is not None else None
        result["recorded_answer_mismatch"] = recorded != parsed
        # TASL may legitimately replace ``trace.answer`` after parsing (for
        # example with a refusal string).  As in the ReAct dialect, retain that
        # mismatch as an audit counter but do not call an otherwise valid raw
        # transcript degenerate.
        result["healthy"] = halted_reason == "final_answer"
        result["reason"] = "ok" if result["healthy"] else "halted_reason"
        return result

    match = FINAL_ANSWER_RE.search(raw)
    if match is None:
        result["reason"] = "missing_final_answer_marker"
        return result
    result["dialect"] = "react"
    parsed = match.group(1).strip()
    result["first_final_payload"] = parsed or None
    trailing_labels = [m.group("label") for m in _TRAILING_PROTOCOL_RE.finditer(raw, match.end())]
    result["trailing_protocol"] = bool(trailing_labels)
    result["trailing_protocol_labels"] = trailing_labels
    result["parsed_answer"] = parsed or None
    if not parsed:
        result["reason"] = "empty_parsed_answer"
        return result
    if _PROTOCOL_PAYLOAD_RE.match(parsed):
        result["parsed_answer"] = None
        result["reason"] = "nested_protocol_payload"
        return result

    recorded = str(answer).strip() if answer is not None else None
    result["recorded_answer_mismatch"] = recorded != parsed

    result["healthy"] = halted_reason == "final_answer"
    result["reason"] = "ok" if result["healthy"] else "halted_reason"
    return result


def final_answer_is_healthy(raw: object, answer: object, halted_reason: object) -> bool:
    """Boolean convenience wrapper around :func:`check_final_answer`."""

    return bool(check_final_answer(raw, answer, halted_reason)["healthy"])
