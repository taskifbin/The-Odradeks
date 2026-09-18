#!/usr/bin/env python3
"""
Unit and Integration Tests for GridWise LLM Energy Optimizer
Can be run standalone or with pytest:
    python tests/test_api.py
    pytest tests/test_api.py
"""

from app.optimizer import optimize_schedule
from app.guardrails import validate_directives
from app.schemas import ScenarioRequest, OptimizeResponse, HealthResponse
from app.main import app
from fastapi.testclient import TestClient
import os
import sys
import json
import unittest
from pathlib import Path

# Add gridwise-llm to python path if not present
REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "gridwise-llm"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


class TestGridWiseService(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.tests_dir = Path(__file__).resolve().parent

    def test_health_endpoint(self):
        """Verify GET /health returns HTTP 200 with status 'ok'."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        # Validate against schema
        HealthResponse(**data)

    def test_optimize_test_json(self):
        """Verify POST /optimize-energy with baseline test.json."""
        test_file = self.tests_dir / "test.json"
        self.assertTrue(test_file.exists(), "test.json not found in tests/")
        with open(test_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Validate request schema
        req = ScenarioRequest(**payload)
        self.assertEqual(len(req.hours), 24)

        # Send request to FastAPI
        response = self.client.post("/optimize-energy", json=payload)
        self.assertEqual(response.status_code, 200, f"Error: {response.text}")
        data = response.json()

        # Validate response schema
        resp_obj = OptimizeResponse(**data)
        self.assertEqual(resp_obj.scenario_id, payload["scenario_id"])
        self.assertEqual(len(resp_obj.hourly_plan), 24)
        self.assertGreater(resp_obj.total_cost_bdt, 0)
        self.assertEqual(len(resp_obj.directive_interpretation),
                         len(payload["operator_notes"]))

    def test_optimize_infeasible_scenario(self):
        """Verify optimize_schedule triggers greedy fallback when constraints are mathematically infeasible."""
        from app.schemas import DirectiveInterpretation, StructuredAdjustment
        infeasible_file = self.tests_dir / "test_infeasible.json"
        self.assertTrue(infeasible_file.exists(),
                        "test_infeasible.json not found in tests/")
        with open(infeasible_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        req = ScenarioRequest(**payload)

        # 1. Directly test optimizer with strictly infeasible directives
        infeasible_directives = [
            DirectiveInterpretation(
                note_index=0,
                applies=True,
                directive_type="max_grid_window",
                structured_adjustment=StructuredAdjustment(
                    hours=list(range(24)),
                    max_grid_kwh=0.0,
                ),
                explanation="Zero grid import allowed.",
            ),
            DirectiveInterpretation(
                note_index=1,
                applies=True,
                directive_type="no_discharge_window",
                structured_adjustment=StructuredAdjustment(
                    hours=list(range(24)),
                ),
                explanation="No battery discharge allowed.",
            ),
        ]

        result = optimize_schedule(
            hours=req.hours,
            battery=req.battery,
            directives=infeasible_directives,
        )
        self.assertIn("Greedy fallback", result["plan_summary"])
        self.assertEqual(len(result["hourly_plan"]), 24)

        # 2. Test endpoint resilience via TestClient
        response = self.client.post("/optimize-energy", json=payload)
        self.assertEqual(response.status_code, 200, f"Error: {response.text}")
        data = response.json()
        resp_obj = OptimizeResponse(**data)
        self.assertEqual(len(resp_obj.hourly_plan), 24)

    def test_guardrails_time_normalization(self):
        """Verify guardrails convert 12-hour AM/PM and range formats to ascending 0-23 hours."""
        from app.schemas import Battery
        battery = Battery(
            capacity_kwh=500,
            initial_energy_kwh=200,
            minimum_energy_kwh=50,
            max_charge_kwh_per_hour=100,
            max_discharge_kwh_per_hour=100,
        )

        raw_llm_output = [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {
                    "hours": [13, 14],
                    "factor": 0.2
                },
                "explanation": "Solar drops to 20% from 1 PM to 3 PM."
            },
            {
                "note_index": 1,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No energy impact."
            }
        ]

        validated = validate_directives(
            raw_interpretations=raw_llm_output,
            battery=battery,
            num_notes=2,
            operator_notes=["Note 1", "Note 2"]
        )
        self.assertEqual(len(validated), 2)
        self.assertEqual(validated[0].directive_type, "solar_reduction")
        self.assertEqual(validated[0].structured_adjustment.hours, [13, 14])
        self.assertEqual(validated[1].directive_type, "no_op")
        self.assertFalse(validated[1].applies)


if __name__ == "__main__":
    unittest.main(verbosity=2)
