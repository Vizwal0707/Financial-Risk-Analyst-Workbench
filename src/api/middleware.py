"""API middleware for request correlation, process timing, and structured error handling."""

import logging
import time
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.serving.feature_builder import ForbiddenLeakageError, MissingFeatureContextError

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injects correlation request ID and calculates wall-clock latency per request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Correlation ID from header or generate new
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        return response


def register_exception_handlers(app: FastAPI) -> None:
    """Registers standard RFC 7807 problem details error handlers."""

    @app.exception_handler(MissingFeatureContextError)
    async def handle_missing_feature_context(request: Request, exc: MissingFeatureContextError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "type": "urn:problem:missing-historical-context",
                "title": "Missing Feature Context",
                "status": 422,
                "detail": str(exc),
                "instance": request_id,
            },
        )

    @app.exception_handler(ForbiddenLeakageError)
    async def handle_forbidden_leakage(request: Request, exc: ForbiddenLeakageError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "type": "urn:problem:data-leakage-violation",
                "title": "Data Leakage Violation",
                "status": 422,
                "detail": str(exc),
                "instance": request_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        errors_summary = []
        is_leakage = False
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            errors_summary.append(f"{loc}: {msg}")
            if "leakage" in msg.lower():
                is_leakage = True

        problem_type = "urn:problem:data-leakage-violation" if is_leakage else "urn:problem:schema-validation-failure"
        problem_title = "Data Leakage Violation" if is_leakage else "Payload Validation Error"

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "type": problem_type,
                "title": problem_title,
                "status": 422,
                "detail": "; ".join(errors_summary),
                "instance": request_id,
                "errors": jsonable_encoder(exc.errors()),
            },
        )
