import re
import math
import logging
from typing import Any, Iterable, Optional

from app.schemas import (
    Battery,
    DirectiveInterpretation,
    StructuredAdjustment,
)

logger = logging.getLogger(__name__)


VALID_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


DIRECTIVE_TYPE_ALIASES = {
    # solar_reduction
    "solar_reduction": "solar_reduction",
    "solar-reduction": "solar_reduction",
    "solar reduction": "solar_reduction",
    "reduce_solar": "solar_reduction",
    "solar_output_reduction": "solar_reduction",
    "pv_reduction": "solar_reduction",
    "panel_reduction": "solar_reduction",
    "solar_drop": "solar_reduction",

    # minimum_battery_reserve
    "minimum_battery_reserve": "minimum_battery_reserve",
    "minimum-battery-reserve": "minimum_battery_reserve",
    "battery_reserve": "minimum_battery_reserve",
    "min_battery_reserve": "minimum_battery_reserve",
    "minimum_reserve": "minimum_battery_reserve",
    "reserve": "minimum_battery_reserve",
    "battery_minimum": "minimum_battery_reserve",
    "min_energy_reserve": "minimum_battery_reserve",

    # no_charge_window
    "no_charge_window": "no_charge_window",
    "no-charge-window": "no_charge_window",
    "no_charge": "no_charge_window",
    "disable_charge": "no_charge_window",
    "charging_disabled": "no_charge_window",
    "charge_disabled": "no_charge_window",
    "no_charging": "no_charge_window",

    # no_discharge_window
    "no_discharge_window": "no_discharge_window",
    "no-discharge-window": "no_discharge_window",
    "no_discharge": "no_discharge_window",
    "disable_discharge": "no_discharge_window",
    "discharging_disabled": "no_discharge_window",
    "discharge_disabled": "no_discharge_window",
    "no_discharging": "no_discharge_window",

    # max_grid_window
    "max_grid_window": "max_grid_window",
    "max-grid-window": "max_grid_window",
    "max_grid": "max_grid_window",
    "grid_cap": "max_grid_window",
    "grid_limit": "max_grid_window",
    "limit_grid": "max_grid_window",
    "maximum_grid": "max_grid_window",

    # no_op
    "no_op": "no_op",
    "no-op": "no_op",
    "noop": "no_op",
    "none": "no_op",
    "irrelevant": "no_op",
    "not_applicable": "no_op",
}


WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty two": 22,
    "twenty-two": 22,
    "twenty-three": 23,
    "twenty three": 23,
}


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _to_float(value: Any) -> Optional[float]:
    """
    Converts common LLM/string representations to float.

    Handles:
    - 20
    - 20.0
    - "20"
    - "20%"
    - "120 kWh"
    - "0.2"
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None

    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return None

        percent = "%" in s

        match = re.search(r"-?\d+(?:\.\d+)?", s)
        if not match:
            return None

        try:
            v = float(match.group())
        except ValueError:
            return None

        if percent:
            v = v / 100.0

        return v if math.isfinite(v) else None

    return None


def _to_int_hour(value: Any) -> Optional[int]:
    """
    Converts a single hour-like value to integer 0-23.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if 0 <= value <= 23 else None

    if isinstance(value, float):
        if value.is_integer():
            iv = int(value)
            return iv if 0 <= iv <= 23 else None
        return None

    if isinstance(value, str):
        return _parse_hour_token(value)

    return None


def _parse_hour_token(token: str) -> Optional[int]:
    """
    Parses strings like:
    - "13"
    - "1 PM"
    - "1pm"
    - "13:00"
    - "one"
    - "three"
    """
    if not isinstance(token, str):
        return None

    s = token.strip().lower()
    if not s:
        return None

    # 24-hour with minutes: 13:00, 09:30
    colon_match = re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", s)
    if colon_match:
        hour = int(colon_match.group(1))
        return hour if 0 <= hour <= 23 else None

    # Numeric hour with optional am/pm
    numeric_match = re.search(
        r"\b(0?[0-9]|1[0-9]|2[0-3])\s*(am|pm|a\.m\.|p\.m\.)?\b",
        s
    )
    if numeric_match:
        hour = int(numeric_match.group(1))
        meridiem = numeric_match.group(2)

        if meridiem:
            meridiem = meridiem.replace(".", "")
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0

        return hour if 0 <= hour <= 23 else None

    # Word numbers
    for word, num in WORD_NUMBERS.items():
        if re.search(rf"\b{re.escape(word)}\b", s):
            return num if 0 <= num <= 23 else None

    return None


def _parse_explicit_time_hours(note: Optional[str]) -> list[int]:
    """
    Conservative deterministic time parser for common explicit patterns:
    - "1 PM to 3 PM"
    - "13:00 to 15:00"
    - "2pm-4pm"
    - "from 6 PM until 9 PM"

    This is only used to repair obvious LLM off-by-one mistakes.
    It does not replace the LLM.
    """
    if not note:
        return []

    text = note.lower()

    # Pattern 1: explicit am/pm pairs
    # Examples:
    # "1 pm to 3 pm"
    # "from 6 pm until 9 pm"
    # "between 2pm and 4pm"
    am_pm_pattern = re.compile(
        r"\b(0?[0-9]|1[0-9]|2[0-3])\s*(am|pm)\b"
        r"\s*(?:to|until|through|and|-|–|—)\s*"
        r"\b(0?[0-9]|1[0-9]|2[0-3])\s*(am|pm)\b",
        re.IGNORECASE,
    )

    match = am_pm_pattern.search(text)
    if match:
        start_hour = _parse_hour_token(f"{match.group(1)} {match.group(2)}")
        end_hour = _parse_hour_token(f"{match.group(3)} {match.group(4)}")

        if start_hour is not None and end_hour is not None and start_hour < end_hour:
            return list(range(start_hour, end_hour))

    # Pattern 2: 24-hour colon pairs
    # Examples:
    # "13:00 to 15:00"
    # "between 14:00 and 16:00"
    colon_pattern = re.compile(
        r"\b([01]?\d|2[0-3]):[0-5]\d\b"
        r"\s*(?:to|until|through|and|-|–|—)\s*"
        r"\b([01]?\d|2[0-3]):[0-5]\d\b",
        re.IGNORECASE,
    )

    match = colon_pattern.search(text)
    if match:
        start_hour = _parse_hour_token(match.group(1) + ":00")
        end_hour = _parse_hour_token(match.group(2) + ":00")

        if start_hour is not None and end_hour is not None and start_hour < end_hour:
            return list(range(start_hour, end_hour))

    # Pattern 3: compact ranges like "1-3 pm", "13-15"
    compact_pm_pattern = re.compile(
        r"\b(0?[0-9]|1[0-9]|2[0-3])\s*(?:-|–|—|to|until|through)\s*"
        r"(0?[0-9]|1[0-9]|2[0-3])\s*(am|pm)\b",
        re.IGNORECASE,
    )

    match = compact_pm_pattern.search(text)
    if match:
        start_raw = match.group(1)
        end_raw = match.group(2)
        meridiem = match.group(3)

        start_hour = _parse_hour_token(f"{start_raw} {meridiem}")
        end_hour = _parse_hour_token(f"{end_raw} {meridiem}")

        if start_hour is not None and end_hour is not None and start_hour < end_hour:
            return list(range(start_hour, end_hour))

    compact_24h_pattern = re.compile(
        r"\b(1[0-9]|2[0-3])\s*(?:-|–|—|to|until|through)\s*"
        r"\b(1[0-9]|2[0-3])\b",
        re.IGNORECASE,
    )

    match = compact_24h_pattern.search(text)
    if match:
        start_hour = _to_int_hour(match.group(1))
        end_hour = _to_int_hour(match.group(2))

        if start_hour is not None and end_hour is not None and start_hour < end_hour:
            return list(range(start_hour, end_hour))

    return []


def _clean_hours(value: Any, note: Optional[str] = None) -> list[int]:
    """
    Deterministically cleans hours into:
    - unique integers
    - 0 through 23
    - ascending order

    Also repairs obvious end-exclusive mistakes using the original note.
    """
    hours: list[int] = []

    # Case: {"start": 13, "end": 15}
    if isinstance(value, dict):
        start = _to_int_hour(value.get("start"))
        end = _to_int_hour(value.get("end"))

        if start is not None and end is not None and start < end:
            hours.extend(range(start, end))
        elif start is not None:
            hours.append(start)

    # Case: string like "13,14" or "13-15"
    elif isinstance(value, str):
        s = value.strip()

        # Range string: "13-15" => [13, 14]
        range_match = re.fullmatch(r"\s*(\d+)\s*(?:-|–|—)\s*(\d+)\s*", s)
        if range_match:
            start = _to_int_hour(range_match.group(1))
            end = _to_int_hour(range_match.group(2))
            if start is not None and end is not None and start < end:
                hours.extend(range(start, end))
            elif start is not None:
                hours.append(start)
        else:
            parts = re.split(r"[,\s;]+", s)
            for part in parts:
                if not part:
                    continue

                inner_range = re.fullmatch(r"(\d+)\s*(?:-|–|—)\s*(\d+)", part)
                if inner_range:
                    start = _to_int_hour(inner_range.group(1))
                    end = _to_int_hour(inner_range.group(2))
                    if start is not None and end is not None and start < end:
                        hours.extend(range(start, end))
                    elif start is not None:
                        hours.append(start)
                else:
                    h = _to_int_hour(part)
                    if h is not None:
                        hours.append(h)

    # Case: list/tuple/set
    elif isinstance(value, Iterable):
        for item in value:
            if isinstance(item, dict):
                start = _to_int_hour(item.get("start"))
                end = _to_int_hour(item.get("end"))
                if start is not None and end is not None and start < end:
                    hours.extend(range(start, end))
                elif start is not None:
                    hours.append(start)
            elif isinstance(item, str):
                inner_range = re.fullmatch(
                    r"\s*(\d+)\s*(?:-|–|—)\s*(\d+)\s*", item)
                if inner_range:
                    start = _to_int_hour(inner_range.group(1))
                    end = _to_int_hour(inner_range.group(2))
                    if start is not None and end is not None and start < end:
                        hours.extend(range(start, end))
                    elif start is not None:
                        hours.append(start)
                else:
                    h = _to_int_hour(item)
                    if h is not None:
                        hours.append(h)
            else:
                h = _to_int_hour(item)
                if h is not None:
                    hours.append(h)

    # Repair using original note if possible
    parsed_hours = _parse_explicit_time_hours(note)

    if parsed_hours:
        if not hours:
            hours = parsed_hours
        else:
            # If LLM included the exclusive end hour, prefer parsed shorter window.
            # Example:
            # note: "1 PM to 3 PM"
            # LLM hours: [13, 14, 15]
            # parsed hours: [13, 14]
            parsed_set = set(parsed_hours)
            hours_set = set(hours)

            if parsed_set.issubset(hours_set) and len(parsed_hours) < len(hours):
                hours = parsed_hours

    clean = sorted({h for h in hours if h is not None and 0 <= h <= 23})
    return clean


def _normalize_directive_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    raw = value.strip().lower()
    if not raw:
        return None

    normalized = raw.replace("-", "_").replace(" ", "_")

    if normalized in VALID_DIRECTIVE_TYPES:
        return normalized

    if raw in DIRECTIVE_TYPE_ALIASES:
        return DIRECTIVE_TYPE_ALIASES[raw]

    if normalized in DIRECTIVE_TYPE_ALIASES:
        return DIRECTIVE_TYPE_ALIASES[normalized]

    return None


def _get_first(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    if not isinstance(mapping, dict):
        return None

    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]

    return None


def _clean_solar_factor(raw_adj: Any) -> Optional[float]:
    """
    Returns remaining usable fraction in [0, 1].

    Handles:
    - factor: 0.2
    - factor: "20%"
    - factor: 20
    - reduction_percent: 80 => factor 0.2
    - reduction_fraction: 0.8 => factor 0.2
    """
    if not isinstance(raw_adj, dict):
        return None

    # Direct remaining-factor keys
    remaining_keys = (
        "factor",
        "remaining_factor",
        "remaining_fraction",
        "usable_fraction",
        "available_fraction",
        "solar_factor",
        "fraction_remaining",
        "remaining",
    )

    for key in remaining_keys:
        v = _to_float(raw_adj.get(key))
        if v is None:
            continue

        if 0.0 <= v <= 1.0:
            return v

        # If model returned 20 meaning 20%
        if 1.0 < v <= 100.0:
            return v / 100.0

    # Reduction keys: invert
    reduction_keys = (
        "reduction",
        "reduction_factor",
        "reduction_fraction",
        "reduction_percent",
        "percent_reduction",
        "cut",
        "cut_fraction",
        "cut_percent",
        "drop",
        "drop_fraction",
        "drop_percent",
    )

    for key in reduction_keys:
        v = _to_float(raw_adj.get(key))
        if v is None:
            continue

        if 0.0 <= v <= 1.0:
            return max(0.0, 1.0 - v)

        if 1.0 < v <= 100.0:
            return max(0.0, 1.0 - (v / 100.0))

    # Generic fallback
    v = _to_float(raw_adj.get("value"))
    if v is not None:
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0

    return None


def _clean_minimum_reserve(raw_adj: Any, battery: Battery) -> Optional[float]:
    """
    Returns finite non-negative reserve not exceeding battery capacity.
    """
    if not isinstance(raw_adj, dict):
        return None

    # Fraction/percentage of capacity keys
    fraction_keys = (
        "reserve_fraction",
        "minimum_fraction",
        "fraction_of_capacity",
        "percent_of_capacity",
        "reserve_percent",
        "minimum_percent",
    )

    for key in fraction_keys:
        v = _to_float(raw_adj.get(key))
        if v is None:
            continue

        if 0.0 <= v <= 1.0:
            reserve = battery.capacity_kwh * v
        elif 1.0 < v <= 100.0:
            reserve = battery.capacity_kwh * (v / 100.0)
        else:
            continue

        reserve = max(0.0, min(reserve, battery.capacity_kwh))
        return round(reserve, 6)

    # Absolute kWh keys
    absolute_keys = (
        "minimum_energy_kwh",
        "reserve_kwh",
        "min_energy_kwh",
        "minimum_kwh",
        "reserve",
        "minimum_energy",
        "battery_reserve_kwh",
        "value",
    )

    v = _get_first(raw_adj, absolute_keys)
    reserve = _to_float(v)

    if reserve is None:
        return None

    if reserve < 0:
        reserve = 0.0

    # Safety clamp to capacity
    reserve = min(reserve, battery.capacity_kwh)

    return round(reserve, 6)


def _clean_max_grid(raw_adj: Any) -> Optional[float]:
    """
    Returns finite non-negative max grid kWh.
    """
    if not isinstance(raw_adj, dict):
        return None

    keys = (
        "max_grid_kwh",
        "grid_cap_kwh",
        "maximum_grid_kwh",
        "max_grid",
        "grid_limit",
        "limit_kwh",
        "value",
    )

    v = _get_first(raw_adj, keys)
    max_grid = _to_float(v)

    if max_grid is None:
        return None

    if max_grid < 0:
        max_grid = 0.0

    return round(max_grid, 6)


def _default_explanation(directive_type: str, note_index: int) -> str:
    if directive_type == "no_op":
        return f"Note {note_index} did not produce a valid supported energy directive after deterministic guardrails."

    return f"Note {note_index} was interpreted as {directive_type} after deterministic validation."


def _make_no_op(note_index: int, explanation: Optional[str] = None) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=explanation or _default_explanation("no_op", note_index),
    )


def validate_directives(
    raw_interpretations: list[dict[str, Any]],
    battery: Battery,
    num_notes: int,
    operator_notes: Optional[list[str]] = None,
) -> list[DirectiveInterpretation]:
    """
    Deterministic guardrails for LLM output.

    Guarantees:
    - exactly one DirectiveInterpretation per operator note
    - note_index order is 0..N-1
    - only supported directive types survive
    - no_op has applies=false and structured_adjustment=null
    - non-no_op has applies=true and valid structured_adjustment
    - hours are unique integers 0-23 ascending
    - numeric values are finite and within allowed ranges
    - malformed LLM output safely becomes no_op
    """
    if operator_notes is None:
        operator_notes = []

    by_index: dict[int, DirectiveInterpretation] = {}
    used_indices: set[int] = set()

    if not isinstance(raw_interpretations, list):
        raw_interpretations = []

    for position, raw in enumerate(raw_interpretations):
        if not isinstance(raw, dict):
            continue

        # ------------------------------------------------------------------
        # Resolve note_index
        # ------------------------------------------------------------------
        note_index = raw.get("note_index")

        if not isinstance(note_index, int) or note_index < 0 or note_index >= num_notes:
            # If note_index is missing/invalid but position is valid, repair by position.
            if position < num_notes and position not in used_indices:
                note_index = position
            else:
                logger.warning("Guardrail: invalid note_index %s at position %s. Skipping.", raw.get(
                    "note_index"), position)
                continue

        if note_index in used_indices:
            logger.warning(
                "Guardrail: duplicate note_index %s. Skipping.", note_index)
            continue

        note_text = operator_notes[note_index] if note_index < len(
            operator_notes) else None

        # ------------------------------------------------------------------
        # Normalize directive type
        # ------------------------------------------------------------------
        directive_type = _normalize_directive_type(raw.get("directive_type"))

        explanation = raw.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = _default_explanation(
                directive_type or "no_op", note_index)

        # ------------------------------------------------------------------
        # no_op path
        # ------------------------------------------------------------------
        if directive_type is None or directive_type == "no_op":
            by_index[note_index] = _make_no_op(note_index, explanation)
            used_indices.add(note_index)
            continue

        raw_adj = raw.get("structured_adjustment")

        # If model says non-no_op but gives no adjustment, try to recover hours from top-level.
        if not isinstance(raw_adj, dict):
            raw_adj = {}

            # Some models wrongly put hours at top level.
            if "hours" in raw:
                raw_adj["hours"] = raw.get("hours")

        hours = _clean_hours(raw_adj.get("hours"), note_text)

        if not hours:
            by_index[note_index] = _make_no_op(
                note_index,
                f"Guardrail rejected {directive_type} for note {note_index} because hours were missing or invalid.",
            )
            used_indices.add(note_index)
            continue

        structured_adjustment: Optional[StructuredAdjustment] = None

        # ------------------------------------------------------------------
        # Directive-specific validation
        # ------------------------------------------------------------------
        if directive_type == "solar_reduction":
            factor = _clean_solar_factor(raw_adj)

            if factor is None:
                by_index[note_index] = _make_no_op(
                    note_index,
                    f"Guardrail rejected solar_reduction for note {note_index} because factor was missing or invalid.",
                )
            else:
                structured_adjustment = StructuredAdjustment(
                    hours=hours,
                    factor=round(float(factor), 6),
                )
                by_index[note_index] = DirectiveInterpretation(
                    note_index=note_index,
                    applies=True,
                    directive_type="solar_reduction",
                    structured_adjustment=structured_adjustment,
                    explanation=explanation,
                )

        elif directive_type == "minimum_battery_reserve":
            minimum_energy = _clean_minimum_reserve(raw_adj, battery)

            if minimum_energy is None:
                by_index[note_index] = _make_no_op(
                    note_index,
                    f"Guardrail rejected minimum_battery_reserve for note {note_index} because minimum_energy_kwh was missing or invalid.",
                )
            else:
                structured_adjustment = StructuredAdjustment(
                    hours=hours,
                    minimum_energy_kwh=minimum_energy,
                )
                by_index[note_index] = DirectiveInterpretation(
                    note_index=note_index,
                    applies=True,
                    directive_type="minimum_battery_reserve",
                    structured_adjustment=structured_adjustment,
                    explanation=explanation,
                )

        elif directive_type == "no_charge_window":
            structured_adjustment = StructuredAdjustment(hours=hours)
            by_index[note_index] = DirectiveInterpretation(
                note_index=note_index,
                applies=True,
                directive_type="no_charge_window",
                structured_adjustment=structured_adjustment,
                explanation=explanation,
            )

        elif directive_type == "no_discharge_window":
            structured_adjustment = StructuredAdjustment(hours=hours)
            by_index[note_index] = DirectiveInterpretation(
                note_index=note_index,
                applies=True,
                directive_type="no_discharge_window",
                structured_adjustment=structured_adjustment,
                explanation=explanation,
            )

        elif directive_type == "max_grid_window":
            max_grid = _clean_max_grid(raw_adj)

            if max_grid is None:
                by_index[note_index] = _make_no_op(
                    note_index,
                    f"Guardrail rejected max_grid_window for note {note_index} because max_grid_kwh was missing or invalid.",
                )
            else:
                structured_adjustment = StructuredAdjustment(
                    hours=hours,
                    max_grid_kwh=max_grid,
                )
                by_index[note_index] = DirectiveInterpretation(
                    note_index=note_index,
                    applies=True,
                    directive_type="max_grid_window",
                    structured_adjustment=structured_adjustment,
                    explanation=explanation,
                )

        else:
            # Should not happen because directive_type was normalized, but keep safe.
            by_index[note_index] = _make_no_op(
                note_index,
                f"Guardrail rejected unsupported directive type for note {note_index}.",
            )

        used_indices.add(note_index)

    # ----------------------------------------------------------------------
    # Guarantee exactly one entry per note
    # ----------------------------------------------------------------------
    for i in range(num_notes):
        if i not in used_indices:
            by_index[i] = _make_no_op(
                i,
                "Guardrail fallback: no valid LLM interpretation was available for this operator note.",
            )

    # ----------------------------------------------------------------------
    # Return in strict note_index order
    # ----------------------------------------------------------------------
    return [by_index[i] for i in range(num_notes)]
