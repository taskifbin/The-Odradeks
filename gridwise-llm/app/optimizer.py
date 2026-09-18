from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any, Dict, List

import pulp

from app.schemas import Battery, DirectiveInterpretation, HourEntry, HourPlan

logger = logging.getLogger(__name__)

_HOURS = range(24)
_EPSILON = 1e-7


def _directive_settings(
    directives: Iterable[Any],
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


def _solve_lp(
    hours: List[HourEntry],
    battery: Battery,
    directives: List[Any],
) -> Dict[str, Any]:
    """Internal LP solver. Raises ValueError if infeasible."""
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

    prob = pulp.LpProblem("gridwise_energy_optimizer", pulp.LpMinimize)
    grid_vars = pulp.LpVariable.dicts("grid", _HOURS, lowBound=0)
    solar_vars = pulp.LpVariable.dicts("solar_used", _HOURS, lowBound=0)
    charge_vars = pulp.LpVariable.dicts("charge", _HOURS, lowBound=0)
    discharge_vars = pulp.LpVariable.dicts("discharge", _HOURS, lowBound=0)
    level_vars = pulp.LpVariable.dicts(
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
        tariff = _number(entry.tariff_bdt_per_kwh,
                         name=f"tariff at hour {hour}")

        if demand < 0 or solar < 0 or tariff < 0:
            raise ValueError(f"hour {hour} contains a negative input")

        prob += solar_vars[hour] <= solar * solar_fraction[hour]
        prob += charge_vars[hour] <= (
            0 if hour in no_charge else battery.max_charge_kwh_per_hour
        )
        prob += discharge_vars[hour] <= (
            0 if hour in no_discharge else battery.max_discharge_kwh_per_hour
        )
        if grid_cap[hour] != float("inf"):
            prob += grid_vars[hour] <= grid_cap[hour]

        prob += (
            solar_vars[hour] + grid_vars[hour] + discharge_vars[hour]
            == demand + charge_vars[hour]
        ), f"energy_balance_{hour}"

        previous_energy = (
            battery.initial_energy_kwh
            if hour == 0
            else level_vars[hour - 1]
        )
        prob += (
            level_vars[hour] == previous_energy +
            charge_vars[hour] - discharge_vars[hour]
        ), f"battery_balance_{hour}"
        prob += level_vars[hour] >= max(
            battery.minimum_energy_kwh,
            reserve[hour],
        )

        objective_terms.append(tariff * grid_vars[hour])
        objective_terms.append(
            _EPSILON * (charge_vars[hour] + discharge_vars[hour]))

    prob += pulp.lpSum(objective_terms)
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    lp_status = pulp.LpStatus[prob.status]
    if lp_status != "Optimal":
        raise ValueError(f"LP Solver failed with status: {lp_status}")

    hourly_plan_list: list[HourPlan] = []
    total_grid_kwh = 0.0
    total_cost_bdt = 0.0
    peak_grid_kwh = 0.0

    tariff_map = {h.hour: h.tariff_bdt_per_kwh for h in hours}

    for h_obj in hours:
        h = h_obj.hour
        g_val = max(0.0, round(float(grid_vars[h].varValue or 0.0), 6))
        s_val = max(0.0, round(float(solar_vars[h].varValue or 0.0), 6))
        c_val = max(0.0, round(float(charge_vars[h].varValue or 0.0), 6))
        d_val = max(0.0, round(float(discharge_vars[h].varValue or 0.0), 6))
        lvl_val = max(0.0, round(float(level_vars[h].varValue or 0.0), 6))

        action = "idle"
        batt_flow = 0.0

        if c_val > 0.001:
            action = "charge"
            batt_flow = c_val
        elif d_val > 0.001:
            action = "discharge"
            batt_flow = d_val

        total_grid_kwh += g_val
        total_cost_bdt += g_val * tariff_map[h]
        if g_val > peak_grid_kwh:
            peak_grid_kwh = g_val

        hourly_plan_list.append(
            HourPlan(
                hour=h,
                grid_kwh=round(g_val, 4),
                solar_used_kwh=round(s_val, 4),
                battery_action=action,
                battery_kwh=round(batt_flow, 4),
                battery_energy_after_kwh=round(lvl_val, 4),
            )
        )

    return {
        "hourly_plan": hourly_plan_list,
        "total_grid_kwh": total_grid_kwh,
        "total_cost_bdt": total_cost_bdt,
        "peak_grid_kwh": peak_grid_kwh,
        "plan_summary": "Optimized via Linear Programming.",
    }


def _greedy_fallback(
    hours: List[HourEntry],
    battery: Battery,
    directives: List[Any],
) -> Dict[str, Any]:
    """Best-effort fallback when LP is infeasible.

    Strategy: Prioritize Solar > Battery Discharge > Grid.
    Try to meet demand within physical limits without crashing.
    """
    logger.warning("Using Greedy Fallback due to LP Infeasibility.")

    solar_factors = {h.hour: 1.0 for h in hours}
    min_reserves = {h.hour: battery.minimum_energy_kwh for h in hours}
    max_grid_caps = {h.hour: float("inf") for h in hours}
    no_charge_hours: set[int] = set()
    no_discharge_hours: set[int] = set()

    for d in directives:
        applies = (
            d.get("applies", False)
            if isinstance(d, dict)
            else getattr(d, "applies", False)
        )
        d_type = (
            d.get("directive_type")
            if isinstance(d, dict)
            else getattr(d, "directive_type", None)
        )
        adj = (
            d.get("structured_adjustment")
            if isinstance(d, dict)
            else getattr(d, "structured_adjustment", None)
        )

        if not applies or d_type == "no_op":
            continue
        if adj is None:
            continue

        if isinstance(adj, dict):
            target_hours = adj.get("hours") or []
            factor = adj.get("factor")
            min_kwh = adj.get("minimum_energy_kwh")
            max_grid = adj.get("max_grid_kwh")
        else:
            target_hours = getattr(adj, "hours", []) or []
            factor = getattr(adj, "factor", None)
            min_kwh = getattr(adj, "minimum_energy_kwh", None)
            max_grid = getattr(adj, "max_grid_kwh", None)

        if d_type == "solar_reduction" and factor is not None:
            for h in target_hours:
                solar_factors[h] = float(factor)
        elif d_type == "minimum_battery_reserve" and min_kwh is not None:
            for h in target_hours:
                min_reserves[h] = max(min_reserves[h], float(min_kwh))
        elif d_type == "no_charge_window":
            no_charge_hours.update(target_hours)
        elif d_type == "no_discharge_window":
            no_discharge_hours.update(target_hours)
        elif d_type == "max_grid_window" and max_grid is not None:
            for h in target_hours:
                max_grid_caps[h] = min(max_grid_caps[h], float(max_grid))

    current_energy = battery.initial_energy_kwh
    hourly_plan_list: list[HourPlan] = []
    total_grid_kwh = 0.0
    total_cost_bdt = 0.0
    peak_grid_kwh = 0.0

    for h_obj in hours:
        h = h_obj.hour
        demand = h_obj.demand_kwh
        tariff = h_obj.tariff_bdt_per_kwh

        # 1. Use Available Solar
        eff_solar = h_obj.solar_kwh * solar_factors[h]
        solar_used = min(demand, eff_solar)
        remaining_demand = demand - solar_used

        # 2. Try Battery Discharge (if allowed and enough energy)
        discharge_allowed = h not in no_discharge_hours
        can_discharge = 0.0

        if discharge_allowed and current_energy > min_reserves[h]:
            available_for_discharge = current_energy - min_reserves[h]
            max_rate = battery.max_discharge_kwh_per_hour
            can_discharge = min(
                remaining_demand, available_for_discharge, max_rate)

        remaining_demand -= can_discharge
        current_energy -= can_discharge

        # 3. Buy from Grid
        grid_cap = max_grid_caps[h]
        grid_buy = min(remaining_demand, grid_cap)
        remaining_demand -= grid_buy

        total_grid_kwh += grid_buy
        total_cost_bdt += grid_buy * tariff
        if grid_buy > peak_grid_kwh:
            peak_grid_kwh = grid_buy

        # Update Energy Level
        charge_amt = 0.0
        start_e = current_energy + can_discharge
        end_e = start_e - can_discharge + charge_amt
        end_e = max(min_reserves[h], min(end_e, battery.capacity_kwh))

        action = "idle"
        batt_flow = 0.0
        if can_discharge > 0.001:
            action = "discharge"
            batt_flow = can_discharge
        elif charge_amt > 0.001:
            action = "charge"
            batt_flow = charge_amt

        hourly_plan_list.append(
            HourPlan(
                hour=h,
                grid_kwh=round(grid_buy, 4),
                solar_used_kwh=round(solar_used, 4),
                battery_action=action,
                battery_kwh=round(batt_flow, 4),
                battery_energy_after_kwh=round(end_e, 4),
            )
        )
        current_energy = end_e

    return {
        "hourly_plan": hourly_plan_list,
        "total_grid_kwh": round(total_grid_kwh, 4),
        "total_cost_bdt": round(total_cost_bdt, 4),
        "peak_grid_kwh": round(peak_grid_kwh, 4),
        "plan_summary": "Greedy fallback applied due to infeasible optimization constraints.",
    }


def optimize_schedule(
    hours: List[HourEntry],
    battery: Battery,
    directives: List[Any],
) -> Dict[str, Any]:
    """Main entry point. Tries LP first, falls back to Greedy if infeasible."""
    try:
        result = _solve_lp(hours, battery, directives)
        result["total_grid_kwh"] = round(result["total_grid_kwh"], 4)
        result["total_cost_bdt"] = round(result["total_cost_bdt"], 4)
        result["peak_grid_kwh"] = round(result["peak_grid_kwh"], 4)
        return result
    except ValueError as e:
        logger.error(f"LP Failed: {e}. Switching to Greedy Fallback.")
        return _greedy_fallback(hours, battery, directives)
    except Exception as e:
        logger.exception("Unexpected error in optimizer")
        raise ValueError(f"Optimizer crashed: {str(e)}")
