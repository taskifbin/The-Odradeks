from typing import Any, Dict, List, Optional

from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

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


@register_validator(name="valid-hour-window", data_type="list")
class ValidHourWindow(Validator):
    """hours must be unique integers 0-23, ascending (Section 05.1)."""

    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if value is None:
            return PassResult()  # no_op entries carry no hours key

        if not isinstance(value, list) or len(value) == 0:
            return FailResult(error_message="hours must be a non-empty array")

        if any(not isinstance(h, int) for h in value):
            return FailResult(error_message="hours entries must be integers")

        if any(h < 0 or h > 23 for h in value):
            return FailResult(error_message="hours entries must be within 0-23")

        if len(set(value)) != len(value):
            return FailResult(
                error_message="hours entries must be unique",
                fix_value=sorted(set(value)),
            )

        if value != sorted(value):
            return FailResult(
                error_message="hours entries must be ascending",
                fix_value=sorted(value),
            )

        return PassResult()


@register_validator(name="valid-directive-type", data_type="string")
class ValidDirectiveType(Validator):
    """directive_type must be one of the six supported values"""

    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        if value not in SUPPORTED_DIRECTIVES:
            return FailResult(
                error_message=(
                    f"'{value}' is not a supported directive_type "
                    f"(must be one of {sorted(SUPPORTED_DIRECTIVES)})"
                ),
                # SAFE FAILURE: fall back to no_op rather than invent a type
                fix_value="no_op",
            )
        return PassResult()


@register_validator(name="solar-factor-range", data_type="float")
class SolarFactorRange(Validator):
    """For solar_reduction, factor must be in [0, 1]."""

    def _validate(self, value: Optional[float], metadata: Dict) -> ValidationResult:
        if value is None:
            return PassResult()
        if not (0 <= value <= 1):
            return FailResult(
                error_message=f"factor {value} out of range [0, 1]",
                fix_value=min(1.0, max(0.0, value)),
            )
        return PassResult()


@register_validator(name="non-negative-number", data_type="float")
class NonNegativeNumber(Validator):
    """minimum_energy_kwh / max_grid_kwh must be finite and >= 0."""

    def _validate(self, value: Optional[float], metadata: Dict) -> ValidationResult:
        if value is None:
            return PassResult()
        try:
            is_bad = value != value or value in (float("inf"), float("-inf"))
        except TypeError:
            return FailResult(error_message=f"{value!r} is not a number")
        if is_bad:
            return FailResult(error_message=f"{value} is not finite")
        if value < 0:
            return FailResult(
                error_message=f"{value} must be non-negative",
                fix_value=0.0,
            )
        return PassResult()


@register_validator(name="applies-no-op-consistency", data_type="dict")
class AppliesNoOpConsistency(Validator):
    """
    Enforces the applies / directive_type / structured_adjustment triangle:

      no_op           -> applies=False, structured_adjustment=None
      everything else -> applies=True, structured_adjustment has required keys
    """

    def _validate(self, value: Dict, metadata: Dict) -> ValidationResult:
        directive_type = value.get("directive_type")
        applies = value.get("applies")
        adjustment = value.get("structured_adjustment")

        if directive_type == "no_op":
            if applies is not False or adjustment is not None:
                return FailResult(
                    error_message="no_op requires applies=false and structured_adjustment=null",
                    fix_value={**value, "applies": False, "structured_adjustment": None},
                )
            return PassResult()

        if directive_type not in REQUIRED_FIELDS_BY_TYPE:
            return PassResult()  # ValidDirectiveType already flags/fixes this case

        if applies is not True:
            return FailResult(
                error_message=f"{directive_type} requires applies=true",
                fix_value={**value, "applies": True},
            )

        if not isinstance(adjustment, dict):
            return FailResult(
                error_message=f"{directive_type} requires a structured_adjustment object"
            )

        missing = [f for f in REQUIRED_FIELDS_BY_TYPE[directive_type] if adjustment.get(f) is None]
        if missing:
            return FailResult(
                error_message=(
                    f"{directive_type} structured_adjustment missing "
                    f"required field(s): {', '.join(missing)}"
                )
            )

        return PassResult()


@register_validator(name="note-index-coverage", data_type="list")
class NoteIndexCoverage(Validator):
    """
    Whole-response check: exactly one entry per operator
    note, note_index covering 0..N-1 with no gaps/duplicates, returned in
    order. Pass num_notes via the validator's metadata at call time.
    """

    def _validate(self, value: List[Dict], metadata: Dict) -> ValidationResult:
        num_notes = metadata.get("num_notes")
        if num_notes is None:
            return FailResult(error_message="num_notes missing from validation metadata")

        indices = [entry.get("note_index") for entry in value]

        if len(indices) != num_notes:
            return FailResult(
                error_message=f"expected {num_notes} entries, got {len(indices)}"
            )

        if sorted(indices) != list(range(num_notes)):
            return FailResult(
                error_message=(
                    f"note_index values must cover 0..{num_notes - 1} exactly "
                    f"once each (got {indices})"
                )
            )

        if indices != list(range(num_notes)):
            return FailResult(
                error_message="entries must be returned in note_index order",
                fix_value=sorted(value, key=lambda e: e["note_index"]),
            )

        return PassResult()