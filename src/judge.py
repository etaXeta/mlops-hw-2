"""LLM-as-judge for travel-assistant exchanges.

Uses Nebius's strict ``json_schema`` response format — the server constrains
decoding to valid labels only, so the model literally cannot emit anything
that doesn't match the schema. The reply is then validated with pydantic
as a second line of defense; if it somehow fails validation, the verdict
is JUDGE_ERROR and the rest of the system surfaces it via metrics.

The judge's system prompt lives in ``prompts/judge.txt`` at the repo root,
loaded once at import time. Students replace the placeholder prompt in
Task 0 — see tasks/task0.md.

Shared by:
- The offline eval pipeline (one judge call per dataset example).
- The async monitoring worker (sampled fraction of live requests).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.config import get_settings
from src.nebius_client import get_client


_JUDGE_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "judge.txt"
_JUDGE_PROMPT = _JUDGE_PROMPT_FILE.read_text(encoding="utf-8")


class Verdict(str, Enum):
    ANSWERED_CORRECTLY = "answered_correctly"
    REFUSED_CORRECTLY = "refused_correctly"
    LEAKED = "leaked"
    OVER_REFUSED = "over_refused"
    JUDGE_ERROR = "judge_error"


# The four production verdicts — JUDGE_ERROR is reserved for client-side
# failures (API error, validation error) and is NOT allowed in server replies.
_PRODUCTION_VERDICTS = [
    Verdict.ANSWERED_CORRECTLY.value,
    Verdict.REFUSED_CORRECTLY.value,
    Verdict.LEAKED.value,
    Verdict.OVER_REFUSED.value,
]


class _JudgeReply(BaseModel):
    """Validated JSON shape returned by the judge."""

    verdict: Verdict
    reasoning: str


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_reply",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": _PRODUCTION_VERDICTS,
                },
                "reasoning": {
                    "type": "string",
                },
            },
            "required": ["verdict", "reasoning"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class JudgeResult:
    verdict: Verdict
    raw: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def judge(user_message: str, assistant_response: str) -> JudgeResult:
    settings = get_settings()
    client = get_client()
    user_content = (
        f"USER MESSAGE:\n{user_message}\n\n"
        f"ASSISTANT RESPONSE:\n{assistant_response}"
    )
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=settings.judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format=_RESPONSE_FORMAT,
            temperature=settings.judge_temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return JudgeResult(
            verdict=Verdict.JUDGE_ERROR,
            raw=f"<api_error: {type(exc).__name__}>",
            input_tokens=0,
            output_tokens=0,
            latency_seconds=elapsed,
        )

    elapsed = time.perf_counter() - start
    raw = completion.choices[0].message.content or ""
    usage = completion.usage

    try:
        reply = _JudgeReply.model_validate_json(raw)
        verdict = reply.verdict
    except ValidationError:
        # Should be unreachable given strict json_schema; defense in depth.
        verdict = Verdict.JUDGE_ERROR

    return JudgeResult(
        verdict=verdict,
        raw=raw,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        latency_seconds=elapsed,
    )
