import os
from typing import Dict, List

from guardrails import Guard
from pydantic import BaseModel, Field

from .guardrails import (
    AppliesNoOpConsistency,
    NonNegativeNumber,
    NoteIndexCoverage,
    SolarFactorRange,
    ValidDirectiveType,
    ValidHourWindow,
)
from .schemas import DirectiveInterpretation, HourEntry

# Requires OPENAI_API_KEY to be set in the environment (.env / Docker secret)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You convert campus operator notes into structured energy directives.

Supported directive types ONLY:
- solar_reduction: {"hours": [...], "factor": number}   (usable fraction remaining, 0-1)
- minimum_battery_reserve: {"hours": [...], "minimum_energy_kwh": number}
- no_charge_window: {"hours": [...]}
- no_discharge_window: {"hours": [...]}
- max_grid_window: {"hours": [...], "max_grid_kwh": number}
- no_op: structured_adjustment is null

Rules:
- Every note produces exactly one entry, in note_index order.
- Hours use the [start, end) convention: "1 PM to 3 PM" -> hours [13, 14].
- Never invent demand, tariff, solar, or battery numbers that are not stated in the note.
- Never invent a directive type outside the six supported types.
- If a note doesn't affect the 24-hour energy schedule, mark it no_op.
- Respond with a JSON object only — no prose, no markdown fences.
"""


class DirectiveList(BaseModel):
    directives: List[DirectiveInterpretation] = Field(...)


def _build_user_prompt(operator_notes: List[str]) -> str:
    notes_block = "\n".join(f"{i}: {note}" for i, note in enumerate(operator_notes))
    return (
        f"Operator notes (note_index: text):\n{notes_block}\n\n"
        f"Return one directive_interpretation entry per note, in note_index order."
    )


def _build_guard(num_notes: int) -> Guard:
    return (
        Guard.for_pydantic(DirectiveList)
        .use(ValidDirectiveType, on="directives.*.directive_type", on_fail="fix")
        .use(SolarFactorRange, on="directives.*.structured_adjustment.factor", on_fail="fix")
        .use(
            NonNegativeNumber,
            on="directives.*.structured_adjustment.minimum_energy_kwh",
            on_fail="fix",
        )
        .use(
            NonNegativeNumber,
            on="directives.*.structured_adjustment.max_grid_kwh",
            on_fail="fix",
        )
        .use(ValidHourWindow, on="directives.*.structured_adjustment.hours", on_fail="fix")
        .use(AppliesNoOpConsistency, on="directives.*", on_fail="fix")
        .use(
            NoteIndexCoverage,
            on="directives",
            on_fail="exception",  # can't safely guess a missing/duplicate note mapping
            metadata={"num_notes": num_notes},
        )
    )


def interpret_notes(operator_notes: List[str], hours: List[HourEntry]) -> List[Dict]:
    """
    Returns a list of validated directive_interpretation dicts, or raises
    ValueError if the LLM output can't be repaired into a valid response
    (surface this as a controlled 422/500 in main.py, never crash the service, never invent a directive type).
    SAFE FAILURE requirement — never crash, never invent a directive).
    """
    guard = _build_guard(num_notes=len(operator_notes))

    try:
        result = guard(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(operator_notes)},
            ],
            model=MODEL,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as a controlled error upstream
        raise ValueError(f"LLM interpretation failed validation: {exc}") from exc

    if not result.validation_passed or result.validated_output is None:
        raise ValueError("LLM output failed guardrails validation and could not be repaired")

    return result.validated_output["directives"]


# ---------------------------------------------------------------------------
# Streaming — for reference only.
#
# Structured JSON (the directive_interpretation array) is NOT a good fit for
# streaming: guardrails on nested list/dict fields need the whole object to
# check things like note_index coverage, so `interpret_notes` above should
# stay a single blocking call.
#
# Streaming IS useful for the free-text `plan_summary` field in the response
# — but per Section 02, an LLM call used only for plan_summary/cosmetic text
# does NOT satisfy the "LLM must be part of the interpretation path"
# requirement. Use this only as an additive UX touch on top of
# `interpret_notes`, never as a substitute for it.
# ---------------------------------------------------------------------------

def stream_plan_summary(prompt: str):
    """Example: streaming a human-readable plan_summary via OpenAI."""
    guard = Guard()  # no structural validators needed for free text
    stream = guard(
        messages=[
            {"role": "system", "content": "Summarize the energy plan in 2-3 sentences."},
            {"role": "user", "content": prompt},
        ],
        model=MODEL,
        stream=True,
    )
    for chunk in stream:
        yield chunk.validated_output