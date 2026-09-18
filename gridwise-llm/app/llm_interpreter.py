import asyncio
import os
import re
import json
import logging
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

from app.schemas import Battery

load_dotenv()
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)


def _get_client() -> AsyncOpenAI:
    load_dotenv()
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=True)

    api_key = os.getenv("PUKU_API_KEY")
    base_url = os.getenv("PUKU_BASE_URL", "https://api.puku.sh/v1")
    timeout = float(os.getenv("PUKU_TIMEOUT_SECONDS", "10.0"))
    max_retries = int(os.getenv("PUKU_MAX_RETRIES", "0"))

    return AsyncOpenAI(
        api_key=api_key or "missing-puku-api-key",
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )


client = _get_client()


SYSTEM_PROMPT = """You are an expert smart-campus energy operator assistant.

Your job is to interpret 1 to 3 natural-language operator notes into structured directives
for a 24-hour energy optimization system.

You must return ONLY valid JSON. Do not return markdown. Do not return explanations outside JSON.

Allowed directive types:
1. solar_reduction
   Meaning: Reduce usable solar during specific hours.
   Required structured_adjustment:
   {
     "hours": [unique integers 0-23 in ascending order],
     "factor": number between 0 and 1
   }

   IMPORTANT: factor is the REMAINING usable fraction.
   Examples:
   - "drop to 20%" => factor = 0.2
   - "80% reduction" => factor = 0.2
   - "one-fifth of normal solar output" => factor = 0.2
   - "reduce by 50%" => factor = 0.5

2. minimum_battery_reserve
   Meaning: Keep battery energy at or above a required level during specific hours.
   Required structured_adjustment:
   {
     "hours": [unique integers 0-23 in ascending order],
     "minimum_energy_kwh": non-negative number
   }

3. no_charge_window
   Meaning: Battery charging is unavailable during specific hours.
   Required structured_adjustment:
   {
     "hours": [unique integers 0-23 in ascending order]
   }

4. no_discharge_window
   Meaning: Battery discharging is unavailable during specific hours.
   Required structured_adjustment:
   {
     "hours": [unique integers 0-23 in ascending order]
   }

5. max_grid_window
   Meaning: Grid import may not exceed a stated amount during specific hours.
   Required structured_adjustment:
   {
     "hours": [unique integers 0-23 in ascending order],
     "max_grid_kwh": non-negative number
   }

6. no_op
   Meaning: The note does not affect the current 24-hour energy schedule.
   Required structured_adjustment: null
   Required applies: false

STRICT TIME RULE:
- Time windows are start-inclusive and end-exclusive.
- "1 PM to 3 PM" means hours [13, 14], not [13, 14, 15].
- "2 PM to 4 PM" means hours [14, 15].
- "6 PM until 9 PM" means hours [18, 19, 20].
- "13:00 to 15:00" means hours [13, 14].

STRICT OUTPUT RULE:
Return a JSON object with exactly one top-level key:

{
  "interpretations": [
    {
      "note_index": 0,
      "applies": true or false,
      "directive_type": "one of the allowed directive types",
      "structured_adjustment": { ... } or null,
      "explanation": "short human-readable reason"
    }
  ]
}

Rules:
- There must be exactly one interpretation object for every operator note.
- note_index must be zero-based and match the input note order.
- For no_op:
  applies must be false
  structured_adjustment must be null
- For every non-no_op directive:
  applies must be true
  structured_adjustment must contain the required fields
- Do not invent unsupported directive types.
- Do not change demand, tariff, battery capacity, base minimum energy, or charge/discharge rates unless a supported directive explicitly allows it.
- If a note is irrelevant, distractor, cosmetic, future-only, unrelated to today's 24-hour energy schedule, or impossible to map to a supported directive, use no_op.
- Be robust to paraphrasing. The same rule may be written in many different ways.

Examples:

Note: "Solar output will drop to about 20% from 1 PM to 3 PM."
Output directive:
{
  "applies": true,
  "directive_type": "solar_reduction",
  "structured_adjustment": {
    "hours": [13, 14],
    "factor": 0.2
  },
  "explanation": "Solar availability is reduced to 20% during the maintenance window."
}

Note: "Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window."
Output directive:
{
  "applies": true,
  "directive_type": "solar_reduction",
  "structured_adjustment": {
    "hours": [13, 14],
    "factor": 0.2
  },
  "explanation": "An 80% reduction means 20% solar output remains."
}

Note: "Do not charge the battery between 2 PM and 4 PM."
Output directive:
{
  "applies": true,
  "directive_type": "no_charge_window",
  "structured_adjustment": {
    "hours": [14, 15]
  },
  "explanation": "Battery charging is disabled from 2 PM to 4 PM."
}

Note: "Keep at least 120 kWh in reserve from 6 PM until 9 PM."
Output directive:
{
  "applies": true,
  "directive_type": "minimum_battery_reserve",
  "structured_adjustment": {
    "hours": [18, 19, 20],
    "minimum_energy_kwh": 120
  },
  "explanation": "Battery reserve must stay at or above 120 kWh during the evening event window."
}

Note: "The cafeteria menu changes tomorrow."
Output directive:
{
  "applies": false,
  "directive_type": "no_op",
  "structured_adjustment": null,
  "explanation": "This note does not affect today's 24-hour energy schedule."
}
"""


def _strip_code_fences(text: str) -> str:
    """
    Some models return:
    ```json
    {...}
    ```
    This removes the fences before JSON parsing.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text,
                      flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def _extract_json(text: str) -> Any:
    """
    Robustly extract JSON from LLM response.
    Handles:
    - pure JSON
    - markdown code fences
    - JSON embedded in extra text
    """
    cleaned = _strip_code_fences(text)

    # First try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start: end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try to find a JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start: end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("LLM response did not contain valid JSON.")


async def _call_llm(messages: list[dict[str, str]], use_json_mode: bool):
    current_client = _get_client()
    kwargs: dict[str, Any] = {
        "model": os.getenv("PUKU_MODEL", "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0.0,
    }

    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return await current_client.chat.completions.create(**kwargs)


async def interpret_notes(operator_notes: list[str], battery: Battery) -> list[dict]:
    """
    Uses puku.sh LLM to interpret operator notes into raw structured dictionaries.

    Returns:
        list[dict]

    Example raw output:
    [
        {
            "note_index": 0,
            "applies": true,
            "directive_type": "solar_reduction",
            "structured_adjustment": {
                "hours": [13, 14],
                "factor": 0.2
            },
            "explanation": "..."
        }
    ]
    """
    if not operator_notes:
        return []

    if not os.getenv("PUKU_API_KEY"):
        raise ValueError("PUKU_API_KEY is not set.")

    user_msg = "Battery context:\n"
    user_msg += f"- capacity_kwh: {battery.capacity_kwh}\n"
    user_msg += f"- initial_energy_kwh: {battery.initial_energy_kwh}\n"
    user_msg += f"- minimum_energy_kwh: {battery.minimum_energy_kwh}\n"
    user_msg += f"- max_charge_kwh_per_hour: {battery.max_charge_kwh_per_hour}\n"
    user_msg += f"- max_discharge_kwh_per_hour: {battery.max_discharge_kwh_per_hour}\n\n"

    user_msg += "Operator notes:\n"
    for i, note in enumerate(operator_notes):
        user_msg += f"[{i}] {note}\n"

    user_msg += "\nReturn ONLY valid JSON with top-level key 'interpretations'."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        try:
            response = await asyncio.wait_for(_call_llm(messages, use_json_mode=True), timeout=10.0)
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            err_text = str(exc).lower()
            # Some providers may not support response_format. Retry once without it.
            if "response_format" in err_text or "json" in err_text:
                logger.warning(
                    "LLM JSON mode failed, retrying without response_format: %s", exc)
                response = await asyncio.wait_for(_call_llm(messages, use_json_mode=False), timeout=10.0)
            else:
                raise

        content = response.choices[0].message.content or ""
        data = _extract_json(content)

        if isinstance(data, dict):
            for key in ("interpretations", "directives", "notes", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return value

            # If model returned a single object instead of array
            if "directive_type" in data or "note_index" in data:
                return [data]

        if isinstance(data, list):
            return data

        raise ValueError("LLM JSON did not contain an interpretations array.")

    except ValueError:
        raise
    except Exception as exc:
        logger.warning("LLM interpretation failed: %s", exc)
        raise ValueError(f"LLM interpretation failed: {exc}") from exc
