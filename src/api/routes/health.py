"""Health and readiness probe endpoints for Kubernetes / container orchestration."""

from datetime import datetime, timezone
from fastapi import APIRouter, Request, Response, status

from src.schemas.response import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health & Governance"])


@router.get("/health", response_model=HealthResponse, summary="Liveness Probe")
async def health_check() -> HealthResponse:
    """Returns basic service liveness status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness Probe")
async def readiness_check(request: Request, response: Response) -> ReadinessResponse:
    """Verifies model loading state, calibrator readiness, and database reachability."""
    state = request.app.state

    models_loaded = getattr(state, "xgb_model", None) is not None and getattr(state, "calibrator", None) is not None
    db_mgr = getattr(state, "db_manager", None)
    db_connected = await db_mgr.ping() if db_mgr is not None else False

    is_ready = models_loaded and db_connected
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    details = {
        "xgboost_loaded": getattr(state, "xgb_model", None) is not None,
        "calibrator_loaded": getattr(state, "calibrator", None) is not None,
        "autoencoder_loaded": getattr(state, "ae_model", None) is not None,
        "explainer_loaded": getattr(state, "explainer_service", None) is not None,
        "database_connected": db_connected,
    }

    return ReadinessResponse(
        status="ready" if is_ready else "degraded",
        models_loaded=models_loaded,
        database_connected=db_connected,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details=details,
    )

