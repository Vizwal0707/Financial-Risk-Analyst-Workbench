"""Robust HTTP client for communicating with the FastAPI Fraud Scoring Engine."""

import logging
from typing import Any, Dict, Optional
import httpx

from dashboard.config.settings import settings

logger = logging.getLogger(__name__)


class FraudApiClient:
    """Client for invoking real-time scoring, governance, and probe endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None) -> None:
        self.base_url = (base_url or settings.api_base_url).rstrip("/")
        self.timeout = timeout or settings.api_timeout_seconds

    def check_health(self) -> Dict[str, Any]:
        """Invokes GET /health."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return {"status": "offline", "error": str(e)}

    def check_ready(self) -> Dict[str, Any]:
        """Invokes GET /ready."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/ready")
                if resp.status_code == 200:
                    return resp.json()
                return {"status": "degraded", "models_loaded": False, "database_connected": False}
        except Exception as e:
            logger.debug(f"Readiness probe failed: {e}")
            return {"status": "offline", "models_loaded": False, "database_connected": False, "error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """Invokes GET /v1/model/info to fetch active lineage and thresholds."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/v1/model/info")
                if resp.status_code == 200:
                    return resp.json()
                return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.warning(f"Failed to fetch model info from API: {e}")
            return {"error": str(e)}

    def score_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches single transaction scoring request to POST /v1/score."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/v1/score", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    data["_success"] = True
                    data["_cached"] = resp.headers.get("X-Cache") == "HIT"
                    return data
                return {
                    "_success": False,
                    "status_code": resp.status_code,
                    "error": resp.text,
                }
        except Exception as e:
            logger.error(f"Transaction scoring request failed: {e}")
            return {"_success": False, "error": str(e)}

    def score_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches vectorized batch scoring request to POST /v1/score/batch."""
        try:
            with httpx.Client(timeout=max(self.timeout * 3.0, 15.0)) as client:
                resp = client.post(f"{self.base_url}/v1/score/batch", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    data["_success"] = True
                    return data
                return {
                    "_success": False,
                    "status_code": resp.status_code,
                    "error": resp.text,
                }
        except Exception as e:
            logger.error(f"Batch scoring request failed: {e}")
            return {"_success": False, "error": str(e)}


api_client = FraudApiClient()

