from app.optimizer import optimize_schedule
from app.guardrails import validate_directives
from app.llm_interpreter import interpret_notes
from app.schemas import (
    ScenarioRequest,
    OptimizeResponse,
    HealthResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import logging
from dotenv import load_dotenv

load_dotenv()


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
    1. Interpret operator notes using LLM.
    2. Validate LLM output using deterministic guardrails.
    3. Optimize 24-hour energy schedule.
    4. Return schema-compliant response.
    """
    try:
        # ------------------------------------------------------------
        # STEP 1: LLM interpretation
        # ------------------------------------------------------------
        # The LLM only needs operator notes and battery context.
        # Do NOT pass hours to the LLM.
        try:
            raw_interpretations = await interpret_notes(
                operator_notes=request.operator_notes,
                battery=request.battery,
            )
        except Exception as exc:
            # Safe fallback: if LLM fails, treat all notes as no_op.
            # This keeps the API alive and still returns a valid schedule.
            logger.warning(
                f"LLM interpretation failed, falling back to no_op: {exc}")
            raw_interpretations = []

        # ------------------------------------------------------------
        # STEP 2: Deterministic guardrails
        # ------------------------------------------------------------
        validated_directives = validate_directives(
            raw_interpretations=raw_interpretations,
            battery=request.battery,
            num_notes=len(request.operator_notes),
        )

        # ------------------------------------------------------------
        # STEP 3: Mathematical optimization
        # ------------------------------------------------------------
        optimization_result = optimize_schedule(
            hours=request.hours,
            battery=request.battery,
            directives=validated_directives,
        )

        # ------------------------------------------------------------
        # STEP 4: Return final response
        # ------------------------------------------------------------
        return OptimizeResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=validated_directives,
            **optimization_result,
        )

    except HTTPException:
        # Already handled HTTP errors, re-raise directly
        raise

    except ValueError as exc:
        # Controlled failure: invalid semantics or infeasible optimization
        logger.warning(f"Validation or optimization error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        # Unexpected internal failure
        logger.exception("Internal optimization failure")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during optimization.",
        ) from exc
