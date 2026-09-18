import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

from app.schemas import Battery

load_dotenv()

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=os.getenv("PUKU_API_KEY", "dummy_key_for_local_test"),
    base_url=os.getenv("PUKU_BASE_URL", "https://api.puku.sh/v1"),
)

MODEL = os.getenv("PUKU_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are an expert energy grid operator assistant. 
Your task is to interpret 1-3 natural-language operator notes into structured directives for a 24-hour energy optimization model.

Supported directive types and their REQUIRED `structured_adjustment` shapes:
1. "solar_reduction": {"hours": [int, ...], "factor": float} 
   - IMPORTANT: 'factor' is the REMAINING usable fraction. (e.g., "80% reduction" -> factor: 0.2).
2. "minimum_battery_reserve": {"hours": [int, ...], "minimum_energy_kwh": float}
3. "no_charge_window": {"hours": [int, ...]}
4. "no_discharge_window": {"hours": [int, ...]}
5. "max_grid_window": {"hours": [int, ...], "max_grid_kwh": float}
6. "no_op": null (Use this ONLY if the note is irrelevant).

STRICT RULES:
- Return a JSON object with a single key "interpretations" containing an array of objects.
- Each object MUST have: "note_index", "applies", "directive_type", "structured_adjustment", "explanation".
- For "no_op", "applies" MUST be false, and "structured_adjustment" MUST be null.
- Time windows are start-inclusive, end-exclusive. "1 PM to 3 PM" means hours [13, 14].
- "hours" arrays must contain unique integers 0-23 in ascending order.
"""


async def interpret_notes(operator_notes: list[str], battery: Battery) -> list[dict]:
    user_msg = "Battery Context:\n"
    user_msg += f"- Capacity: {battery.capacity_kwh} kWh\n"
    user_msg += f"- Base Minimum: {battery.minimum_energy_kwh} kWh\n\n"

    user_msg += "Operator Notes:\n"
    for i, note in enumerate(operator_notes):
        user_msg += f"[{i}] {note}\n"

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=15.0,
        )

        raw_content = response.choices[0].message.content
        if not raw_content:
            raise ValueError("LLM returned empty content")

        raw_data = json.loads(raw_content)

        if isinstance(raw_data, dict) and "interpretations" in raw_data:
            return raw_data["interpretations"]
        elif isinstance(raw_data, list):
            return raw_data
        else:
            raise ValueError(
                f"Unexpected LLM JSON structure: {type(raw_data)}")

    except Exception as e:
        logger.error(f"LLM interpretation failed: {e}")
        raise ValueError(f"LLM interpretation failed: {str(e)}")
