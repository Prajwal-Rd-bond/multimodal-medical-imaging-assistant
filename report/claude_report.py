"""LLM formatting layer -- two Claude API calls per the plan:

  1. Raw pipeline outputs -> validated structured JSON (tool use, forces the
     exact schema instead of trusting free-text parsing).
  2. Structured JSON -> prose report, with a mandatory disclaimer template
     and uncertainty-aware language.

Requires ANTHROPIC_API_KEY in the environment. Both calls are pure
functions over a StructuredFindings/dict so they're easy to unit test with a
stubbed client (see tests/test_smoke.py).
"""
from __future__ import annotations

import json
import os

import anthropic

from report.schema import FINDINGS_TOOL_SCHEMA, StructuredFindings

DISCLAIMER_TEMPLATE = (
    "This is a research demo, not a diagnostic device. It has not been "
    "reviewed or approved for clinical use. Do not use this output to make "
    "or inform any medical decision. Always consult a qualified healthcare "
    "professional."
)


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY to use the Claude report layer.")
    return anthropic.Anthropic(api_key=api_key)


def validate_findings_via_tool_use(
    raw_pipeline_output: dict,
    client: anthropic.Anthropic | None = None,
    model: str = "claude-sonnet-5",
) -> StructuredFindings:
    """Call #1: force the raw pipeline dict through the findings tool schema.

    This is deliberately not free-text parsing -- `tool_choice` forces
    Claude to call `emit_structured_findings`, so the response is guaranteed
    to be valid JSON matching FINDINGS_TOOL_SCHEMA before it ever reaches
    our pydantic model.
    """
    client = client or get_client()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[FINDINGS_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "emit_structured_findings"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Normalize this medical imaging pipeline output into the "
                    "emit_structured_findings tool call exactly. Do not add, "
                    "remove, or alter any findings or numeric values -- only "
                    "reshape the data:\n\n" + json.dumps(raw_pipeline_output)
                ),
            }
        ],
    )
    tool_call = next(b for b in response.content if b.type == "tool_use")
    return StructuredFindings.model_validate(tool_call.input)


REPORT_SYSTEM_PROMPT = """You write structured-findings summaries for a research demo of a \
medical imaging assistant. You are NOT a clinician and this is NOT a diagnostic tool.

Rules, all mandatory:
- Begin the report with the disclaimer verbatim, exactly as given.
- Never use definitive diagnostic language ("this is", "the patient has"). Use \
  hedged language ("the model flags a finding consistent with...", "possible...").
- If `ood_flag` is true, or a finding has `flag_uncertain: true`, explicitly say the \
  findings for that item are inconclusive / low-confidence and should not be relied upon.
- Keep it to one short paragraph per finding plus a one-line overall summary. No lists of \
  next steps, no treatment suggestions.
- Output plain prose only, no markdown headers."""


def generate_prose_report(
    findings: StructuredFindings,
    client: anthropic.Anthropic | None = None,
    model: str = "claude-sonnet-5",
    disclaimer: str = DISCLAIMER_TEMPLATE,
) -> str:
    """Call #2: structured JSON -> prose report."""
    client = client or get_client()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=REPORT_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Disclaimer to use verbatim as the first line:\n{disclaimer}\n\n"
                    f"Structured findings JSON:\n{findings.model_dump_json(indent=2)}"
                ),
            }
        ],
    )
    text_block = next(b for b in response.content if b.type == "text")
    report = text_block.text
    if disclaimer.strip()[:30] not in report:
        # Belt-and-braces: never ship a report missing the disclaimer, even
        # if the model drifted from instructions.
        report = f"{disclaimer}\n\n{report}"
    return report


def generate_full_report(
    raw_pipeline_output: dict,
    client: anthropic.Anthropic | None = None,
    model: str = "claude-sonnet-5",
    disclaimer: str = DISCLAIMER_TEMPLATE,
    skip_validation_call: bool = False,
) -> tuple[StructuredFindings, str]:
    """Runs both calls back to back. `skip_validation_call=True` lets callers
    who already have a trustworthy StructuredFindings (e.g. built directly
    via report.build_findings, not LLM-derived) skip straight to the prose
    call -- call #1 only adds value when the input isn't already validated
    pydantic data."""
    client = client or get_client()
    if skip_validation_call:
        findings = StructuredFindings.model_validate(raw_pipeline_output)
    else:
        findings = validate_findings_via_tool_use(raw_pipeline_output, client, model)
    report = generate_prose_report(findings, client, model, disclaimer)
    return findings, report
