import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

from app.schemas import Battery

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Puku.sh client
client = AsyncOpenAI(
    api_key=os.getenv("PUKU_API_KEY", "dummy_key_for_local_test"),
    base_url=os.getenv("PUKU_BASE_URL", "https://api.puku.sh/v1"),
)

MODEL = os.getenv("PUKU_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are an expert energy grid operator assistant. 
Your task is to interpret 1-3 natural-language operator notes into structured directives for a 24-hour energy optimization model.

Supported directive types and their REQUIRED `structured_adjustment` shapes:
1. "solar_reduction": {"hours": [int, ...], "factor": float} 
   - IMPORTANT: 'factor' is the REMAINING usable fraction. (e.g., "80% reduction" or "drop to 20%" -> factor: 0.2).
2. "minimum_battery_reserve": {"hours": [int, ...], "minimum_energy_kwh": float}
3. "no_charge_window": {"hours": [int, ...]}
4. "no_discharge_window": {"hours": [int, ...]}
5. "max_grid_window": {"hours": [int, ...], "max_grid_kwh": float}
6. "no_op": null (Use this ONLY if the note is irrelevant to the 24-hour energy schedule, e.g., "cafeteria menu changes").

STRICT RULES:
- You MUST return a JSON object with a single key "interpretations" containing an array of objects.
- Each object in the array MUST have exactly these keys: 
  "note_index" (0-based integer matching the input note), 
  "applies" (boolean), 
  "directive_type" (string, one of the 6 types above), 
  "structured_adjustment" (object or null), 
  "explanation" (string, brief reason).
- For "no_op", "applies" MUST be false, and "structured_adjustment" MUST be null.
- For all other directives, "applies" MUST be true.
- Time windows are start-inclusive, end-exclusive. "1 PM to 3 PM" means hours [13, 14].
- "hours" arrays must contain unique integers from 0 to 23 in strictly ascending order.
- Do NOT invent directives, demand, or battery limits not supported above.
"""


async def interpret_notes(operator_notes: list[str], battery: Battery) -> list[dict]:
    user_msg = "Battery Context (for reference when interpreting numeric values):\n"
    user_msg += f"- Total Capacity: {battery.capacity_kwh} kWh\n"
    user_msg += f"- Base Minimum Reserve: {battery.minimum_energy_kwh} kWh\n"
    user_msg += f"- Max Charge/Discharge Rate: {battery.max_charge_kwh_per_hour} kWh/hr\n\n"

    user_msg += "Operator Notes to Interpret:\n"
    for i, note in enumerate(operator_notes):
        user_msg += f"[{i}] {note}\n"

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},  # Forces root to be {}
            temperature=0.0,  # Deterministic output
            timeout=15.0,  # Allow enough time for LLM to respond
        )

        raw_content = response.choices[0].message.content
        if not raw_content:
            raise ValueError("LLM returned empty content")

        raw_data = json.loads(raw_content)

        if isinstance(raw_data, dict) and "interpretations" in raw_data:
            return raw_data["interpretations"]
        elif isinstance(raw_data, list):
            # Fallback if the LLM ignored the "interpretations" key instruction
            return raw_data
        else:
            raise ValueError(
                f"Unexpected LLM JSON structure: {type(raw_data)}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {e}")
        raise ValueError(
            "LLM returned invalid JSON. Please check the prompt or model.")
    except Exception as e:
        logger.error(f"LLM interpretation failed: {e}")
        raise ValueError(f"LLM interpretation failed: {str(e)}")
