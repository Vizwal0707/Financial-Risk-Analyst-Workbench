"""FastAPI application factory, middleware registration, and router assembly."""

import logging
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml

from src.api.lifespan import lifespan
from src.api.middleware import RequestContextMiddleware, register_exception_handlers
from src.api.routes.health import router as health_router
from src.api.routes.info import router as info_router
from src.api.routes.score import router as score_router

logger = logging.getLogger(__name__)


def create_app(config_path: str = "configs/api_config.yaml") -> FastAPI:
    """Creates and configures the FastAPI fraud scoring intelligence service."""
    # Load configuration
    cfg = {}
    p = Path(config_path)
    if p.exists():
        try:
            with open(p, "r") as f:
                cfg = yaml.safe_load(f).get("api", {})
        except Exception as e:
            logger.warning(f"Failed to read API config from {config_path}: {e}")

    title = cfg.get("title", "Financial Fraud Detection & Risk Intelligence API")
    version = cfg.get("version", "1.0.0")
    description = cfg.get(
        "description",
        "High-throughput inference, scoring, and explainability REST API for payment transaction monitoring.",
    )

    app = FastAPI(
        title=title,
        version=version,
        description=description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. CORS Configuration (Environment Overridable for Hardened Deployments)
    cors_cfg = cfg.get("cors", {})
    env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
    if env_origins:
        allowed_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    else:
        allowed_origins = cors_cfg.get("allow_origins", ["*"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=cors_cfg.get("allow_methods", ["GET", "POST", "OPTIONS"]),
        allow_headers=cors_cfg.get("allow_headers", ["*"]),
    )

    # 2. Correlation ID & Latency Timing Middleware
    app.add_middleware(RequestContextMiddleware)

    # 3. Standardized RFC 7807 Error Handlers
    register_exception_handlers(app)

    # 4. Mount API Routers
    app.include_router(health_router)
    app.include_router(info_router)
    app.include_router(score_router)

    return app


# Default ASGI application instance for Uvicorn
app = create_app()

