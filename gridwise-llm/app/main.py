import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ScenarioRequest,
    OptimizeResponse,
    HealthResponse,
)
from app.llm_interpreter import interpret_notes
from app.optimizer import optimize_schedule

logger = logging.getLogger(__name__)

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
def optimize_energy(request: ScenarioRequest):
    """
    Main endpoint:
    1. Interprets operator notes via LLM (guardrails run inline, Section 08).
    2. Optimizes the 24-hour energy schedule (Section 05-09).

    NOTE: defined as a plain `def`, not `async def`. interpret_notes() makes
    a blocking network call to the LLM provider — FastAPI automatically runs
    sync path operations in a threadpool, which keeps that call from
    blocking the event loop without needing `await`/asyncio plumbing here.
    """
    try:
        # STEP 1: LLM Interpretation + inline guardrails (Section 03, 08)
        # interpret_notes() already runs the Guard(...).use(...) validators
        # from guardrails.py before returning — its output is trusted.
        validated_directives = interpret_notes(
            operator_notes=request.operator_notes,
            hours=request.hours,
        )

        # STEP 2: Mathematical Optimization (Section 05-09)
        optimization_result = optimize_schedule(
            hours=request.hours,
            battery=request.battery,
            directives=validated_directives,
        )

        # STEP 3: Construct and Return Final Response
        return OptimizeResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=validated_directives,
            **optimization_result,
        )

    except ValueError as e:
        # Malformed/unrecoverable LLM output, or an infeasible schedule —
        # both are "controlled" failures per Section 08's SAFE FAILURE rule.
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.exception("Internal optimization failure")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during optimization.",
        ) from e