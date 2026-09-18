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
import asyncio
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
    """Readiness endpoint for the judging harness.

    Must return {"status": "ok"} with HTTP 200.
    """
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeResponse, tags=["Optimization"])
async def optimize_energy(request: ScenarioRequest):
    try:
        # STEP 1: LLM Interpretation with STRICT TIMEOUT and FALLBACK
        raw_interpretations = []
        try:
            raw_interpretations = await interpret_notes(
                operator_notes=request.operator_notes,
                battery=request.battery,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM Timed Out. Falling back to no_op.")
            raw_interpretations = []
        except Exception as exc:
            logger.warning(f"LLM Error: {exc}. Falling back to no_op.")
            raw_interpretations = []

        # STEP 2: Deterministic Guardrails
        # Even if raw_interpretations is [], guardrails will fill no_ops
        validated_directives = validate_directives(
            raw_interpretations=raw_interpretations,
            battery=request.battery,
            num_notes=len(request.operator_notes),
            operator_notes=request.operator_notes,
        )

        # STEP 3: Optimization (with internal Greedy Fallback)
        optimization_result = optimize_schedule(
            hours=request.hours,
            battery=request.battery,
            directives=validated_directives,
        )

        # STEP 4: Response Construction
        # Ensure floats are rounded to 4 decimals to prevent schema validation issues
        # and match judge tolerance expectations
        resp_dict = {
            "scenario_id": request.scenario_id,
            "directive_interpretation": validated_directives,
            **optimization_result,
        }

        return OptimizeResponse(**resp_dict)

    except HTTPException:
        raise
    except ValueError as exc:
        # Catch specific validation/infeasibility errors if they bubble up
        logger.warning(f"Validation Error: {exc}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Critical Internal Error")
        # Do NOT expose stack trace to client
        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        ) from exc
