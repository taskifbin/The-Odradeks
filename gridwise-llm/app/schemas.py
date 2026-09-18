from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


#request

class HourEntry(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class Battery(BaseModel):
    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_bounds(self) -> "Battery":
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh cannot exceed capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError("initial_energy_kwh cannot be below minimum_energy_kwh")
        return self


class ScenarioRequest(BaseModel):
    scenario_id: str = Field(..., min_length=1)
    operator_notes: list[str] = Field(..., min_length=1, max_length=3)
    hours: list[HourEntry] = Field(..., min_length=24, max_length=24)
    battery: Battery

    @field_validator("operator_notes")
    @classmethod
    def notes_non_empty(cls, notes: list[str]) -> list[str]:
        for n in notes:
            if not n.strip():
                raise ValueError("operator_notes entries must be non-empty strings")
        return notes

    @field_validator("hours")
    @classmethod
    def hours_cover_0_to_23(cls, hours: list[HourEntry]) -> list[HourEntry]:
        seen = [h.hour for h in hours]
        if sorted(seen) != list(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0..23")
        return hours


# ---------------------------------------------------------------------------
# Directive interpretation schema (Section 04, 05, 08)
# ---------------------------------------------------------------------------

class StructuredAdjustment(BaseModel):
    """
    Union-style payload. Which fields are populated depends on directive_type:

      solar_reduction          -> hours, factor
      minimum_battery_reserve  -> hours, minimum_energy_kwh
      no_charge_window         -> hours
      no_discharge_window      -> hours
      max_grid_window          -> hours, max_grid_kwh
      no_op                    -> not used (structured_adjustment is null instead)
    """

    hours: Optional[list[int]] = None
    factor: Optional[float] = Field(default=None, ge=0, le=1)
    minimum_energy_kwh: Optional[float] = Field(default=None, ge=0)
    max_grid_kwh: Optional[float] = Field(default=None, ge=0)

    @field_validator("hours")
    @classmethod
    def hours_unique_ascending_in_range(
        cls, hours: Optional[list[int]]
    ) -> Optional[list[int]]:
        if hours is None:
            return hours
        if len(hours) == 0:
            raise ValueError("hours array must not be empty")
        if any(h < 0 or h > 23 for h in hours):
            raise ValueError("hours entries must be integers 0..23")
        if len(set(hours)) != len(hours):
            raise ValueError("hours entries must be unique")
        if hours != sorted(hours):
            raise ValueError("hours entries must be in ascending order")
        return hours


class DirectiveInterpretation(BaseModel):
    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[StructuredAdjustment]
    explanation: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_applies_semantics(self) -> "DirectiveInterpretation":
        # no_op <-> applies=False, structured_adjustment=None (Section 05.1, 08)
        if self.directive_type == "no_op":
            if self.applies is not False:
                raise ValueError("no_op directives must have applies = false")
            if self.structured_adjustment is not None:
                raise ValueError("no_op directives must have structured_adjustment = null")
            return self

        if self.applies is not True:
            raise ValueError(
                f"{self.directive_type} directives must have applies = true"
            )
        if self.structured_adjustment is None:
            raise ValueError(
                f"{self.directive_type} directives require a structured_adjustment"
            )

        adj = self.structured_adjustment
        required = {
            "solar_reduction": ("hours", "factor"),
            "minimum_battery_reserve": ("hours", "minimum_energy_kwh"),
            "no_charge_window": ("hours",),
            "no_discharge_window": ("hours",),
            "max_grid_window": ("hours", "max_grid_kwh"),
        }[self.directive_type]

        for field_name in required:
            if getattr(adj, field_name) is None:
                raise ValueError(
                    f"{self.directive_type} requires '{field_name}' in structured_adjustment"
                )

        return self


#respones

class HourPlan(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0)
    solar_used_kwh: float = Field(..., ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(..., ge=0)
    battery_energy_after_kwh: float = Field(..., ge=0)

    @model_validator(mode="after")
    def idle_means_zero(self) -> "HourPlan":
        if self.battery_action == "idle" and self.battery_kwh != 0:
            raise ValueError("battery_kwh must be 0 when battery_action is idle")
        return self


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan] = Field(..., min_length=24, max_length=24)
    total_grid_kwh: float = Field(..., ge=0)
    total_cost_bdt: float = Field(..., ge=0)
    peak_grid_kwh: float = Field(..., ge=0)
    plan_summary: str

    @field_validator("hourly_plan")
    @classmethod
    def plan_covers_0_to_23(cls, plan: list[HourPlan]) -> list[HourPlan]:
        hours = [p.hour for p in plan]
        if sorted(hours) != list(range(24)):
            raise ValueError("hourly_plan must contain exactly one entry per hour 0..23")
        return plan


#health response 

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"