from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ScenarioRequest,
    OptimizeResponse,
    HealthResponse,
)
from app.llm_interpreter import interpret_notes
from app.guardrails import validate_directives
from app.optimizer import optimize_schedule

app = FastAPI(
    title="GridWise LLM Energy Optimizer",
    description="BUP CSE Fest 2026 - Smart Campus Energy Optimization Challenge",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Readiness endpoint for the judging harness.
    Must return {"status": "ok"} with HTTP 200.
    """
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeResponse, tags=["Optimization"])
async def optimize_energy(request: ScenarioRequest):
    """
    Main endpoint: 
    1. Interprets operator notes via LLM.
    2. Validates with deterministic guardrails.
    3. Optimizes the 24-hour energy schedule.
    """
    try:
        # STEP 1: LLM Interpretation

        raw_interpretations = await interpret_notes(
            operator_notes=request.operator_notes,
            battery=request.battery
        )

        # STEP 2: Deterministic Guardrails
        # Cleans, validates, and enforces strict rules on the LLM's raw output
        validated_directives = validate_directives(
            raw_interpretations=raw_interpretations,
            battery=request.battery,
            num_notes=len(request.operator_notes)
        )

        # STEP 3: Mathematical Optimization
        # Solves the Linear Programming problem to minimize grid cost
        optimization_result = optimize_schedule(
            hours=request.hours,
            battery=request.battery,
            directives=validated_directives
        )

        # STEP 4: Construct and Return Final Response
        return OptimizeResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=validated_directives,

            **optimization_result
        )

    except ValueError as e:

        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:

        print(f"[ERROR] Internal optimization failure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during optimization."
        )
