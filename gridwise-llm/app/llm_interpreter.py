from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .schemas import Battery

logger = logging.getLogger(__name__)

MODEL = os.getenv("PUKU_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

client = AsyncOpenAI(
    api_key=os.getenv("PUKU_API_KEY", "dummy_key_for_local_test"),
    base_url=os.getenv("PUKU_BASE_URL", "https://api.puku.sh/v1"),
)

SYSTEM_PROMPT = """You convert campus operator notes into structured energy directives.

Supported directive types ONLY:
- solar_reduction: {"hours": [...], "factor": number}   (usable fraction remaining, 0 to 1)
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
Format:
{
  "directives": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
      "explanation": "Solar output will drop to 20% from 1 PM to 3 PM."
    }
  ]
}
"""


def _build_user_prompt(operator_notes: List[str]) -> str:
    notes_block = "\n".join(f"{i}: {note}" for i,
                            note in enumerate(operator_notes))
    return (
        f"Operator notes (note_index: text):\n{notes_block}\n\n"
        f"Return one directive_interpretation entry per note, in note_index order as JSON."
    )


def _parse_hour_window(text: str) -> List[int]:
    """Parse time window phrases like '1 PM to 3 PM' into [start, end) hour integers."""
    t = text.lower()
    # Match patterns like '1 pm to 3 pm', 'between 2 pm and 4 pm'
    m = re.search(
        r"(?:from|between)?\s*(\d{1,2})(?::00)?\s*(am|pm)?\s*(?:to|and|-)\s*(\d{1,2})(?::00)?\s*(am|pm)",
        t,
    )
    if m:
        h1_str, p1_str, h2_str, p2_str = m.groups()
        h1, h2 = int(h1_str), int(h2_str)
        p2 = p2_str.lower()
        p1 = p1_str.lower() if p1_str else p2

        if p1 == "pm" and h1 < 12:
            h1 += 12
        if p1 == "am" and h1 == 12:
            h1 = 0
        if p2 == "pm" and h2 < 12:
            h2 += 12
        if p2 == "am" and h2 == 12:
            h2 = 0

        if h1 < h2:
            return list(range(h1, h2))
        elif h1 > h2:
            return list(range(h1, 24)) + list(range(0, h2))

    # Match simple hour ranges like "hours 13 to 15"
    m2 = re.search(r"hours?\s*(\d{1,2})\s*(?:to|-)\s*(\d{1,2})", t)
    if m2:
        h1, h2 = int(m2.group(1)), int(m2.group(2))
        if 0 <= h1 < h2 <= 24:
            return list(range(h1, h2))

    return []


def _heuristic_fallback(operator_notes: List[str]) -> List[Dict[str, Any]]:
    """Heuristic interpreter for notes if the LLM provider is unreachable."""
    results: List[Dict[str, Any]] = []
    for idx, note in enumerate(operator_notes):
        lower = note.lower()
        hours = _parse_hour_window(note)

        # 1. Solar reduction
        if "solar" in lower and any(w in lower for w in ["drop", "reduc", "curtail", "cut"]):
            m_to = re.search(
                r"(?:drop|fall|cut|curtail|reduc\w*)\s+to\s+(\d+(?:\.\d+)?)\s*%", lower)
            m_by = re.search(
                r"(?:drop|fall|cut|curtail|reduc\w*)\s+by\s+(\d+(?:\.\d+)?)\s*%", lower)
            m_gen = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)

            if m_to:
                factor = float(m_to.group(1)) / 100.0
            elif m_by:
                factor = max(0.0, 1.0 - float(m_by.group(1)) / 100.0)
            elif m_gen:
                factor = float(m_gen.group(1)) / 100.0
            else:
                factor = 0.5

            factor = max(0.0, min(1.0, factor))
            if not hours:
                hours = [11, 12, 13, 14]

            results.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": hours, "factor": factor},
                "explanation": note,
            })
            continue

        # 2. No charge window
        if any(p in lower for p in ["no charge", "do not charge", "don't charge", "stop charge", "stop charging"]):
            results.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {"hours": hours if hours else [14, 15]},
                "explanation": note,
            })
            continue

        # 3. No discharge window
        if any(p in lower for p in ["no discharge", "do not discharge", "don't discharge", "stop discharge"]):
            results.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "no_discharge_window",
                "structured_adjustment": {"hours": hours if hours else [18, 19]},
                "explanation": note,
            })
            continue

        # 4. Minimum battery reserve
        if any(p in lower for p in ["reserve", "minimum battery", "min battery"]):
            m_kwh = re.search(r"(\d+(?:\.\d+)?)\s*kwh", lower)
            val = float(m_kwh.group(1)) if m_kwh else 50.0
            results.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {"hours": hours if hours else list(range(24)), "minimum_energy_kwh": val},
                "explanation": note,
            })
            continue

        # 5. Max grid window
        if "grid" in lower and any(p in lower for p in ["max", "cap", "limit"]):
            m_kwh = re.search(r"(\d+(?:\.\d+)?)\s*kwh", lower)
            val = float(m_kwh.group(1)) if m_kwh else 100.0
            results.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": hours if hours else list(range(24)), "max_grid_kwh": val},
                "explanation": note,
            })
            continue

        # 6. No-op (e.g. cafeteria menu, weather commentary, greetings)
        results.append({
            "note_index": idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": note,
        })

    return results


async def interpret_notes(
    operator_notes: List[str],
    battery: Optional[Battery] = None,
    hours: Optional[Any] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Interpret campus operator notes into structured energy directives.

    Calls the LLM provider via AsyncOpenAI. Falls back gracefully to heuristic
    parsing if network or provider issues occur.
    """
    directives: List[Dict[str, Any]] = []

    try:
        user_prompt = _build_user_prompt(operator_notes)
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        if isinstance(data, dict):
            raw_directives = data.get(
                "directives", data.get("directive_interpretation", []))
            if isinstance(raw_directives, list):
                directives = raw_directives
            elif isinstance(raw_directives, dict):
                directives = [raw_directives]
        elif isinstance(data, list):
            directives = data

        if not directives:
            logger.warning(
                "LLM returned empty directives; using heuristic fallback")
            directives = _heuristic_fallback(operator_notes)

    except Exception as exc:
        logger.warning(
            f"LLM API call failed ({exc}); falling back to heuristic parsing")
        directives = _heuristic_fallback(operator_notes)

    return directives


async def stream_plan_summary(prompt: str):
    """Example: streaming a human-readable plan_summary via OpenAI."""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Summarize the energy plan in 2-3 sentences."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                yield delta
    except Exception as exc:
        logger.warning(f"Streaming failed: {exc}")
        yield "Optimized energy plan summary."
