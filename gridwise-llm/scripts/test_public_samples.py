#!/usr/bin/env python3
"""
Test script to validate GridWise LLM Energy Optimizer against public test scenarios.
Usage:
    python scripts/test_public_samples.py [base_url]
Default base_url: http://localhost:8000
"""

import sys
import os
import json
import urllib.request
import urllib.error

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("API_URL", "http://localhost:8000")
    base_url = base_url.rstrip("/")

    print(f"[*] Testing target: {base_url}")

    # 1. Health check
    health_url = f"{base_url}/health"
    print(f"\n--- [1] Checking Health Endpoint: {health_url} ---")
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "GridWise-Test"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            print(f"[+] Status Code: {resp.status}")
            print(f"[+] Response: {body}")
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            assert body.get("status") == "ok", f"Expected status 'ok', got {body}"
            print("[✓] Health check PASSED")
    except Exception as e:
        print(f"[✗] Health check FAILED: {e}")
        sys.exit(1)

    # 2. Test sample scenarios
    test_files = ["test.json", "test_infeasible.json"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [os.getcwd(), script_dir, os.path.join(script_dir, ".."), os.path.join(script_dir, "..", "..")]

    for test_file in test_files:
        filepath = None
        for d in search_dirs:
            candidate = os.path.join(d, test_file)
            if os.path.isfile(candidate):
                filepath = candidate
                break

        if not filepath:
            print(f"\n[!] Notice: {test_file} not found in search paths, skipping.")
            continue

        print(f"\n--- [2] Testing Optimization with {test_file} ({filepath}) ---")
        with open(filepath, "r", encoding="utf-8") as f:
            payload_data = json.load(f)

        optimize_url = f"{base_url}/optimize-energy"
        req_data = json.dumps(payload_data).encode("utf-8")
        req = urllib.request.Request(
            optimize_url,
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "GridWise-Test"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                resp_bytes = resp.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))
                print(f"[+] Status Code: {resp.status}")
                print(f"[+] Scenario ID: {resp_json.get('scenario_id')}")
                print(f"[+] Total Cost: {resp_json.get('total_cost_bdt')} BDT")
                print(f"[+] Total Grid: {resp_json.get('total_grid_kwh')} kWh")
                print(f"[+] Peak Grid: {resp_json.get('peak_grid_kwh')} kWh")

                plan = resp_json.get("hourly_plan", [])
                print(f"[+] Hourly plan entries returned: {len(plan)}")
                assert len(plan) == 24, f"Expected 24 hours, got {len(plan)}"

                initial_battery = payload_data.get("battery", {}).get("initial_energy_kwh", 0)
                final_battery = plan[-1].get("battery_energy_after_kwh", -1) if plan else -1
                print(f"[+] Battery neutrality: initial={initial_battery} kWh, final={final_battery} kWh")

                interpretations = resp_json.get("directive_interpretation", [])
                print(f"[+] Directives interpreted: {len(interpretations)}")
                for item in interpretations:
                    print(f"    - Note {item.get('note_index')}: {item.get('directive_type')} (applies={item.get('applies')})")
                    if item.get("structured_adjustment"):
                        print(f"      Adjustment: {item.get('structured_adjustment')}")

                print(f"[✓] {test_file} optimization test PASSED")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            print(f"[✗] Optimization failed with HTTP {e.code}: {err_body}")
            sys.exit(1)
        except Exception as e:
            print(f"[✗] Optimization failed: {e}")
            sys.exit(1)

    print("\n==========================================")
    print("  ALL VALIDATION TESTS PASSED SUCCESSFULLY! ")
    print("==========================================")

if __name__ == "__main__":
    main()
