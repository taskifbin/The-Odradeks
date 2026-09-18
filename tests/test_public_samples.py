#!/usr/bin/env python3
"""
GridWise Test Suite - Public Sample Scenarios Validator
Validates live FastAPI service against all scenarios in the tests directory.

Usage:
    python tests/test_public_samples.py [base_url]
    # Or from inside the tests directory:
    python test_public_samples.py [base_url]

Default base_url: http://localhost:8000
"""

import os
import sys
import json
import glob
import urllib.request
import urllib.error


def validate_scenario_response(payload: dict, resp_json: dict) -> list[str]:
    """Validate physical constraints and schema compliance of response."""
    errors = []

    # 1. Schema fields
    required_fields = [
        "scenario_id", "directive_interpretation", "hourly_plan",
        "total_grid_kwh", "total_cost_bdt", "peak_grid_kwh", "plan_summary"
    ]
    for rf in required_fields:
        if rf not in resp_json:
            errors.append(f"Missing required field '{rf}' in response")

    # 2. Hourly plan structure
    plan = resp_json.get("hourly_plan", [])
    if len(plan) != 24:
        errors.append(f"Expected 24 hour plans, got {len(plan)}")

    # 3. Hours coverage 0..23
    hours_seen = [p.get("hour") for p in plan]
    if sorted(hours_seen) != list(range(24)):
        errors.append(f"Hourly plan hours must strictly cover 0..23 in order")

    # 4. Physical checks per hour
    battery_info = payload.get("battery", {})
    cap = battery_info.get("capacity_kwh", 0)
    min_energy = battery_info.get("minimum_energy_kwh", 0)

    for i, p in enumerate(plan):
        hour = p.get("hour")
        grid = p.get("grid_kwh", 0)
        solar_used = p.get("solar_used_kwh", 0)
        action = p.get("battery_action")
        batt_kwh = p.get("battery_kwh", 0)
        energy_after = p.get("battery_energy_after_kwh", 0)

        if grid < 0:
            errors.append(f"Hour {hour}: negative grid_kwh {grid}")
        if solar_used < 0:
            errors.append(f"Hour {hour}: negative solar_used_kwh {solar_used}")
        if energy_after < min_energy - 1e-3:
            errors.append(
                f"Hour {hour}: battery energy {energy_after} below minimum {min_energy}")
        if energy_after > cap + 1e-3:
            errors.append(
                f"Hour {hour}: battery energy {energy_after} above capacity {cap}")
        if action == "idle" and batt_kwh > 1e-4:
            errors.append(
                f"Hour {hour}: battery is idle but battery_kwh is {batt_kwh}")

    # 5. Directives
    req_notes = payload.get("operator_notes", [])
    interps = resp_json.get("directive_interpretation", [])
    if len(interps) != len(req_notes):
        errors.append(
            f"Expected {len(req_notes)} interpretations, got {len(interps)}")

    return errors


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv(
        "API_URL", "http://localhost:8000")
    base_url = base_url.rstrip("/")

    print("=" * 65)
    print(f"  GridWise Energy Optimizer - Public Samples Test Runner")
    print(f"  Target Endpoint: {base_url}")
    print("=" * 65)

    # 1. Health Check
    health_url = f"{base_url}/health"
    print(f"\n[1/2] Checking Health Endpoint: {health_url}")
    try:
        req = urllib.request.Request(
            health_url, headers={"User-Agent": "GridWise-Test"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if resp.status == 200 and body.get("status") == "ok":
                print(
                    f"  [PASS] Status: {body.get('status')} (HTTP {resp.status})")
            else:
                print(
                    f"  [FAIL] Unexpected response: {body} (HTTP {resp.status})")
                sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] Health check failed: {e}")
        print(
            "  -> Ensure your application is running (e.g. uvicorn app.main:app or Docker)")
        sys.exit(1)

    # 2. Discover test files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [script_dir, os.getcwd(
    ), os.path.join(script_dir, "..", "tests")]
    json_files = []
    for d in candidate_dirs:
        found = sorted(glob.glob(os.path.join(d, "test*.json")))
        for f in found:
            basename = os.path.basename(f)
            if not any(os.path.basename(existing) == basename for existing in json_files):
                json_files.append(f)

    if not json_files:
        print("\n[!] No test*.json files found in search directories.")
        sys.exit(1)

    print(
        f"\n[2/2] Running Optimization Tests against {len(json_files)} scenario(s)...")

    all_passed = True
    for test_file in json_files:
        fname = os.path.basename(test_file)
        print(f"\n--- Testing Scenario: {fname} ---")
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  [FAIL] Could not read {fname}: {e}")
            all_passed = False
            continue

        optimize_url = f"{base_url}/optimize-energy"
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            optimize_url,
            data=req_data,
            headers={"Content-Type": "application/json",
                     "User-Agent": "GridWise-Test"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw_bytes = resp.read()
                resp_json = json.loads(raw_bytes.decode("utf-8"))

                print(f"  Scenario ID:    {resp_json.get('scenario_id')}")
                print(
                    f"  Total Cost:     {resp_json.get('total_cost_bdt'):.2f} BDT")
                print(
                    f"  Total Grid:     {resp_json.get('total_grid_kwh'):.2f} kWh")
                print(
                    f"  Peak Grid:      {resp_json.get('peak_grid_kwh'):.2f} kWh")
                print(f"  Plan Summary:   {resp_json.get('plan_summary')}")

                # Check directives
                interps = resp_json.get("directive_interpretation", [])
                print(f"  Directives ({len(interps)} interpreted):")
                for item in interps:
                    adj_str = f" | {item.get('structured_adjustment')}" if item.get(
                        "structured_adjustment") else ""
                    print(
                        f"    - Note {item.get('note_index')}: {item.get('directive_type')} (applies={item.get('applies')}){adj_str}")

                # Validate physics & schema
                errors = validate_scenario_response(payload, resp_json)
                if errors:
                    print(f"  [FAIL] Physics / Schema Validation Errors:")
                    for err in errors:
                        print(f"    * {err}")
                    all_passed = False
                else:
                    print(
                        f"  [PASS] {fname} passed all physical and schema checks.")

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            print(f"  [FAIL] HTTP Error {e.code}: {err_body}")
            all_passed = False
        except Exception as e:
            print(f"  [FAIL] Request failed: {e}")
            all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("  🎉 ALL TEST SCENARIOS PASSED PERFECTLY!")
    else:
        print("  ❌ SOME TESTS FAILED. See details above.")
    print("=" * 65)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
