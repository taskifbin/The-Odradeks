from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import Battery, DirectiveInterpretation, StructuredAdjustment

logger = logging.getLogger(__name__)

SUPPORTED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

# Required structured_adjustment keys per directive_type
REQUIRED_FIELDS_BY_TYPE = {
    "solar_reduction": ("hours", "factor"),
    "minimum_battery_reserve": ("hours", "minimum_energy_kwh"),
    "no_charge_window": ("hours",),
    "no_discharge_window": ("hours",),
    "max_grid_window": ("hours", "max_grid_kwh"),
}


class ValidationResult:
    def __init__(self, passed: bool, error_message: Optional[str] = None, fix_value: Any = None):
        self.passed = passed
        self.error_message = error_message
        self.fix_value = fix_value


class PassResult(ValidationResult):
    def __init__(self):
        super().__init__(passed=True)


class FailResult(ValidationResult):
    def __init__(self, error_message: str, fix_value: Any = None):
        super().__init__(passed=False, error_message=error_message, fix_value=fix_value)


class Validator:
    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        return PassResult()


class ValidHourWindow(Validator):
    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if value is None:
            return PassResult()
        if not isinstance(value, list) or len(value) == 0:
            return FailResult(error_message="hours must be a non-empty array")
        if any(not isinstance(h, int) for h in value):
            return FailResult(error_message="hours entries must be integers")
        if any(h < 0 or h > 23 for h in value):
            return FailResult(error_message="hours entries must be within 0-23")
        if len(set(value)) != len(value):
            return FailResult(error_message="hours entries must be unique", fix_value=sorted(set(value)))
        if value != sorted(value):
            return FailResult(error_message="hours entries must be ascending", fix_value=sorted(value))
        return PassResult()


class ValidDirectiveType(Validator):
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        if value not in SUPPORTED_DIRECTIVES:
            return FailResult(
                error_message=f"'{value}' is not a supported directive_type",
                fix_value="no_op",
            )
        return PassResult()


class SolarFactorRange(Validator):
    def _validate(self, value: Optional[float], metadata: Dict) -> ValidationResult:
        if value is None:
            return PassResult()
        if not (0 <= value <= 1):
            return FailResult(
                error_message=f"factor {value} out of range [0, 1]",
                fix_value=min(1.0, max(0.0, float(value))),
            )
        return PassResult()


class NonNegativeNumber(Validator):
    def _validate(self, value: Optional[float], metadata: Dict) -> ValidationResult:
        if value is None:
            return PassResult()
        try:
            val = float(value)
            is_bad = val != val or val in (float("inf"), float("-inf"))
        except (TypeError, ValueError):
            return FailResult(error_message=f"{value!r} is not a number")
        if is_bad:
            return FailResult(error_message=f"{value} is not finite")
        if val < 0:
            return FailResult(error_message=f"{val} must be non-negative", fix_value=0.0)
        return PassResult()


class AppliesNoOpConsistency(Validator):
    def _validate(self, value: Dict, metadata: Dict) -> ValidationResult:
        directive_type = value.get("directive_type")
        applies = value.get("applies")
        adjustment = value.get("structured_adjustment")
        if directive_type == "no_op":
            if applies is not False or adjustment is not None:
                return FailResult(
                    error_message="no_op requires applies=false and structured_adjustment=null",
                    fix_value={**value, "applies": False,
                               "structured_adjustment": None},
                )
            return PassResult()
        if directive_type not in REQUIRED_FIELDS_BY_TYPE:
            return PassResult()
        if applies is not True:
            return FailResult(
                error_message=f"{directive_type} requires applies=true",
                fix_value={**value, "applies": True},
            )
        if not isinstance(adjustment, dict):
            return FailResult(error_message=f"{directive_type} requires a structured_adjustment object")
        missing = [f for f in REQUIRED_FIELDS_BY_TYPE[directive_type]
                   if adjustment.get(f) is None]
        if missing:
            return FailResult(error_message=f"{directive_type} missing: {missing}")
        return PassResult()


class NoteIndexCoverage(Validator):
    def _validate(self, value: List[Dict], metadata: Dict) -> ValidationResult:
        num_notes = metadata.get("num_notes")
        if num_notes is None:
            return FailResult(error_message="num_notes missing")
        indices = [entry.get("note_index") for entry in value]
        if len(indices) != num_notes or sorted(indices) != list(range(num_notes)):
            return FailResult(error_message=f"Indices must cover 0..{num_notes - 1}")
        return PassResult()


def _clean_single_directive(
    item: Any,
    note_idx: int,
    battery: Optional[Battery] = None,
) -> Dict[str, Any]:
    """Validate and sanitize a single directive entry, guaranteeing valid schema output."""
    if not isinstance(item, dict):
        return {
            "note_index": note_idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "Invalid directive payload",
        }

    directive_type = item.get("directive_type")
    raw_explanation = item.get("explanation")
    explanation = str(raw_explanation).strip(
    ) if raw_explanation else "Directive interpreted."
    if not explanation:
        explanation = "Directive interpreted."

    if directive_type not in SUPPORTED_DIRECTIVES or directive_type == "no_op":
        return {
            "note_index": note_idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": explanation,
        }

    adj = item.get("structured_adjustment")
    if not isinstance(adj, dict):
        return {
            "note_index": note_idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": explanation,
        }

    raw_hours = adj.get("hours")
    if not isinstance(raw_hours, list) or len(raw_hours) == 0:
        return {
            "note_index": note_idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": explanation,
        }

    valid_hours: list[int] = []
    for h in raw_hours:
        try:
            h_int = int(h)
            if 0 <= h_int <= 23:
                valid_hours.append(h_int)
        except (ValueError, TypeError):
            continue

    valid_hours = sorted(set(valid_hours))
    if not valid_hours:
        return {
            "note_index": note_idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": explanation,
        }

    clean_adj: Dict[str, Any] = {"hours": valid_hours}

    if directive_type == "solar_reduction":
        try:
            factor = float(adj.get("factor", 1.0))
            clean_adj["factor"] = max(0.0, min(1.0, factor))
        except (ValueError, TypeError):
            clean_adj["factor"] = 1.0

    elif directive_type == "minimum_battery_reserve":
        try:
            min_energy = float(adj.get("minimum_energy_kwh", 0.0))
            if battery is not None:
                clean_adj["minimum_energy_kwh"] = min(
                    float(battery.capacity_kwh), max(0.0, min_energy))
            else:
                clean_adj["minimum_energy_kwh"] = max(0.0, min_energy)
        except (ValueError, TypeError):
            clean_adj["minimum_energy_kwh"] = 0.0

    elif directive_type == "max_grid_window":
        try:
            max_grid = float(adj.get("max_grid_kwh", 0.0))
            clean_adj["max_grid_kwh"] = max(0.0, max_grid)
        except (ValueError, TypeError):
            clean_adj["max_grid_kwh"] = 0.0

    # Build Pydantic model to verify strict schema validity
    try:
        model = DirectiveInterpretation(
            note_index=note_idx,
            applies=True,
            directive_type=directive_type,
            structured_adjustment=StructuredAdjustment(**clean_adj),
            explanation=explanation,
        )
        return model.model_dump(exclude_unset=True)
    except Exception as exc:
        logger.warning(
            f"Failed to validate directive {note_idx} with Pydantic: {exc}")
        return {
            "note_index": note_idx,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": explanation,
        }


def validate_directives(
    raw_interpretations: Any,
    battery: Optional[Battery] = None,
    num_notes: int = 1,
) -> List[Dict[str, Any]]:
    """Deterministic guardrails: enforce schema, note coverage, and safe bounds.

    Guarantees:
    1. Returns exactly `num_notes` directives for note_index 0..num_notes-1 in order.
    2. Enforces valid directive_type, hour windows, and bounds.
    3. Gracefully falls back to no_op if any directive cannot be safely repaired.
    """
    items_by_index: Dict[int, Any] = {}
    if isinstance(raw_interpretations, list):
        for idx, item in enumerate(raw_interpretations):
            if isinstance(item, dict):
                note_idx = item.get("note_index")
                if isinstance(note_idx, int) and 0 <= note_idx < num_notes:
                    if note_idx not in items_by_index:
                        items_by_index[note_idx] = item
                elif idx < num_notes and idx not in items_by_index:
                    items_by_index[idx] = item

    validated: List[Dict[str, Any]] = []
    for i in range(num_notes):
        item = items_by_index.get(i)
        if item is not None:
            clean = _clean_single_directive(item, note_idx=i, battery=battery)
        else:
            clean = {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No directive provided for note",
            }
        validated.append(clean)

    return validated
