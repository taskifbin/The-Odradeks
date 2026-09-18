from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

import pulp

from .schemas import Battery, HourEntry

_HOURS = range(24)
_EPSILON = 1e-7


def _directive_settings(
    directives: Iterable[Mapping[str, Any]],
) -> tuple[dict[int, float], set[int], set[int], dict[int, float], dict[int, float]]:
    """Convert validated directive payloads into per-hour LP limits."""
    solar_fraction = {hour: 1.0 for hour in _HOURS}
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    reserve = {hour: 0.0 for hour in _HOURS}
    grid_cap = {hour: float("inf") for hour in _HOURS}

    for directive in directives:
        applies = (
            directive.get("applies", False)
            if isinstance(directive, dict)
            else getattr(directive, "applies", False)
        )
        if not applies:
            continue

        raw_adj = (
            directive.get("structured_adjustment")
            if isinstance(directive, dict)
            else getattr(directive, "structured_adjustment", None)
        )
        if raw_adj is None:
            continue

        if isinstance(raw_adj, dict):
            adjustment = raw_adj
        elif hasattr(raw_adj, "model_dump"):
            adjustment = raw_adj.model_dump()
        else:
            adjustment = getattr(raw_adj, "__dict__", {})

        hours = adjustment.get("hours") or []
        directive_type = (
            directive.get("directive_type")
            if isinstance(directive, dict)
            else getattr(directive, "directive_type", None)
        )

        if directive_type == "solar_reduction":
            factor = adjustment.get("factor")
            if factor is not None:
                for hour in hours:
                    solar_fraction[hour] *= float(factor)
        elif directive_type == "minimum_battery_reserve":
            minimum = adjustment.get("minimum_energy_kwh")
            if minimum is not None:
                for hour in hours:
                    reserve[hour] = max(reserve[hour], float(minimum))
        elif directive_type == "no_charge_window":
            no_charge.update(hours)
        elif directive_type == "no_discharge_window":
            no_discharge.update(hours)
        elif directive_type == "max_grid_window":
            maximum = adjustment.get("max_grid_kwh")
            if maximum is not None:
                for hour in hours:
                    grid_cap[hour] = min(grid_cap[hour], float(maximum))

    return solar_fraction, no_charge, no_discharge, reserve, grid_cap


def _number(value: float | None, *, name: str) -> float:
    if value is None or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def optimize_schedule(
    hours: list[HourEntry],
    battery: Battery,
    directives: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Optimize a 24-hour energy schedule using a linear PuLP model.

    Grid imports are minimized by tariff-weighted cost. Solar may be curtailed,
    while battery charge/discharge and state-of-charge limits are enforced for
    every hour. A small cycling penalty only breaks otherwise equivalent LP
    solutions and does not materially change the cost objective.
    """
    if len(hours) != 24 or {entry.hour for entry in hours} != set(_HOURS):
        raise ValueError(
            "hours must contain exactly one entry for each hour 0..23")

    by_hour = {entry.hour: entry for entry in hours}
    (
        solar_fraction,
        no_charge,
        no_discharge,
        reserve,
        grid_cap,
    ) = _directive_settings(directives)

    problem = pulp.LpProblem("gridwise_energy_optimizer", pulp.LpMinimize)
    grid = pulp.LpVariable.dicts("grid", _HOURS, lowBound=0)
    solar_used = pulp.LpVariable.dicts("solar_used", _HOURS, lowBound=0)
    charge = pulp.LpVariable.dicts("charge", _HOURS, lowBound=0)
    discharge = pulp.LpVariable.dicts("discharge", _HOURS, lowBound=0)
    battery_energy = pulp.LpVariable.dicts(
        "battery_energy",
        _HOURS,
        lowBound=0,
        upBound=battery.capacity_kwh,
    )

    objective_terms: list[pulp.LpAffineExpression] = []
    for hour in _HOURS:
        entry = by_hour[hour]
        demand = _number(entry.demand_kwh, name=f"demand at hour {hour}")
        solar = _number(entry.solar_kwh, name=f"solar at hour {hour}")
        tariff = _number(
            entry.tariff_bdt_per_kwh,
            name=f"tariff at hour {hour}",
        )
        if demand < 0 or solar < 0 or tariff < 0:
            raise ValueError(f"hour {hour} contains a negative input")

        problem += solar_used[hour] <= solar * solar_fraction[hour]
        problem += charge[hour] <= (
            0 if hour in no_charge else battery.max_charge_kwh_per_hour
        )
        problem += discharge[hour] <= (
            0 if hour in no_discharge else battery.max_discharge_kwh_per_hour
        )
        if grid_cap[hour] != float("inf"):
            problem += grid[hour] <= grid_cap[hour]

        problem += (
            solar_used[hour] + grid[hour] + discharge[hour]
            == demand + charge[hour]
        ), f"energy_balance_{hour}"

        previous_energy = (
            battery.initial_energy_kwh
            if hour == 0
            else battery_energy[hour - 1]
        )
        problem += (
            battery_energy[hour] == previous_energy +
            charge[hour] - discharge[hour]
        ), f"battery_balance_{hour}"
        problem += battery_energy[hour] >= max(
            battery.minimum_energy_kwh,
            reserve[hour],
        )

        objective_terms.append(tariff * grid[hour])
        objective_terms.append(_EPSILON * (charge[hour] + discharge[hour]))

    problem += pulp.lpSum(objective_terms)
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(
            f"energy schedule is infeasible (solver status: {pulp.LpStatus[status]})"
        )

    plans: list[dict[str, Any]] = []
    for hour in _HOURS:
        grid_value = max(0.0, float(pulp.value(grid[hour]) or 0.0))
        solar_value = max(0.0, float(pulp.value(solar_used[hour]) or 0.0))
        charge_value = max(0.0, float(pulp.value(charge[hour]) or 0.0))
        discharge_value = max(0.0, float(pulp.value(discharge[hour]) or 0.0))
        energy_value = max(0.0, float(pulp.value(battery_energy[hour]) or 0.0))

        if charge_value > _EPSILON:
            action = "charge"
            battery_value = charge_value
        elif discharge_value > _EPSILON:
            action = "discharge"
            battery_value = discharge_value
        else:
            action = "idle"
            battery_value = 0.0

        plans.append(
            {
                "hour": hour,
                "grid_kwh": grid_value,
                "solar_used_kwh": solar_value,
                "battery_action": action,
                "battery_kwh": battery_value,
                "battery_energy_after_kwh": energy_value,
            }
        )

    total_grid = sum(plan["grid_kwh"] for plan in plans)
    total_cost = sum(
        plan["grid_kwh"] * by_hour[plan["hour"]].tariff_bdt_per_kwh
        for plan in plans
    )
    peak_grid = max(plan["grid_kwh"] for plan in plans)
    return {
        "hourly_plan": plans,
        "total_grid_kwh": total_grid,
        "total_cost_bdt": total_cost,
        "peak_grid_kwh": peak_grid,
        "plan_summary": (
            f"Optimized 24-hour schedule using {total_grid:.2f} kWh from the grid "
            f"at a total cost of {total_cost:.2f} BDT; peak grid import was "
            f"{peak_grid:.2f} kWh."
        ),
    }
