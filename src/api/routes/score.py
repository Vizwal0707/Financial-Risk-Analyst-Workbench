"""High-performance real-time fraud scoring endpoints for single transactions and batches."""

import json
import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Response, status
import numpy as np

from src.schemas.response import (
    BatchScoreResponse,
    BatchScoreSummary,
    FraudExplanationResponse,
    FraudScoreResponse,
)
from src.schemas.transaction import (
    BatchTransactionInput,
    TransactionInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Real-Time Scoring"])


@router.post("/score", response_model=FraudScoreResponse, summary="Score Single Transaction")
async def score_transaction(
    payload: TransactionInput,
    request: Request,
    response: Response,
) -> FraudScoreResponse:
    """Evaluates a single live transaction, returning calibrated risk score, triage decision, and audit trail."""
    state = request.app.state
    t0 = time.perf_counter()

    # 1. State assertions
    xgb_model = getattr(state, "xgb_model", None)
    calibrator = getattr(state, "calibrator", None)
    feature_builder = getattr(state, "feature_builder", None)
    policy_engine = getattr(state, "policy_engine", None)
    audit_repo = getattr(state, "audit_repo", None)
    explainer_service = getattr(state, "explainer_service", None)

    if xgb_model is None or calibrator is None or feature_builder is None or policy_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine models are not loaded into application state.",
        )

    # 2. Idempotency Check: deduplicate requests by request_id
    effective_req_id = payload.get_effective_request_id()
    if audit_repo is not None:
        cached_record = await audit_repo.get_by_request_id(effective_req_id)
        if cached_record is not None:
            response.headers["X-Cache"] = "HIT"
            cached_explanation = None
            if cached_record.get("explanation"):
                cached_explanation = FraudExplanationResponse(**cached_record["explanation"])

            return FraudScoreResponse(
                request_id=cached_record["request_id"],
                transaction_id=f"{cached_record['name_orig']}_{cached_record['step']}",
                raw_probability=cached_record["raw_probability"],
                calibrated_probability=cached_record["calibrated_probability"],
                risk_score=cached_record["risk_score"],
                risk_tier=cached_record["risk_tier"],
                decision=cached_record["decision"],
                operational_recommendation=cached_record["action_code"],
                action_code=cached_record["action_code"],
                decision_reasons=cached_record["decision_reasons"],
                risk_weighted_investigation_priority=cached_record.get("risk_weighted_investigation_priority"),
                triage_score=cached_record.get("risk_weighted_investigation_priority"),
                anomaly_score=cached_record.get("anomaly_score"),
                latency_ms=cached_record["latency_ms"],
                model_version=cached_record["model_version"],
                cold_start_applied=bool(cached_record.get("cold_start_applied", 0)),
                simulated_recommendation_only=True,
                recommendation_disclaimer=(
                    "Simulated operational recommendation for risk evaluation only; "
                    "prohibited from triggering real payment authorization, settlement, or authentication workflows."
                ),
                explanation=cached_explanation,
            )

    response.headers["X-Cache"] = "MISS"

    # 3. Feature transformation & validation
    X_single, cold_start_applied = feature_builder.transform_single(payload)

    # 4. Primary Model Scoring
    raw_prob = float(xgb_model.predict_proba(X_single)[:, 1][0])

    # 5. Probability Calibration
    calibrated_prob = float(calibrator.calibrate([raw_prob])[0])

    # 6. Optional Deep Autoencoder Anomaly Signal
    anomaly_score: Optional[float] = None
    ae_model = getattr(state, "ae_model", None)
    ae_scaler = getattr(state, "ae_scaler", None)
    if ae_model is not None and ae_scaler is not None:
        try:
            X_scaled = ae_scaler.transform(X_single)
            if hasattr(ae_model, "compute_reconstruction_errors"):
                mse_errors, _ = ae_model.compute_reconstruction_errors(X_scaled)
            else:
                mse_errors, _ = ae_model.compute_reconstruction_error(X_scaled)
            anomaly_score = float(ae_model.predict_anomaly_score(mse_errors)[0])
        except Exception as e:
            logger.debug(f"Autoencoder anomaly evaluation failed: {e}")

    # 7. Decision Policy Evaluation
    feats_dict, _ = feature_builder._build_features_dict(payload)
    decision_res = policy_engine.evaluate_decision(
        calibrated_prob=calibrated_prob,
        amount=payload.amount,
        features_dict=feats_dict,
        anomaly_score=anomaly_score,
        cold_start_applied=cold_start_applied,
    )

    # 8. Explainability: Generate SHAP card if requested or on REVIEW / DECLINE
    explanation_resp: Optional[FraudExplanationResponse] = None
    if (payload.include_explanation or decision_res.decision in ("REVIEW", "DECLINE")) and explainer_service is not None:
        try:
            explanation_resp = explainer_service.explain_transaction(
                features_row=X_single[0],
                predicted_prob=calibrated_prob,
                decision=decision_res.decision,
                anomaly_score=anomaly_score,
            )
        except Exception as e:
            logger.warning(f"Failed to generate SHAP explanation: {e}")

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    model_ver = getattr(state, "model_version", "1.0.0")

    # 9. Asynchronous Audit Persistence (Fault-Tolerant)
    if audit_repo is not None:
        expl_json = explanation_resp.model_dump_json() if explanation_resp else None
        try:
            await audit_repo.log_decision(
                request_id=effective_req_id,
                step=payload.step,
                corridor_type=payload.type,
                amount=payload.amount,
                name_orig=payload.nameOrig,
                oldbalance_orig=payload.oldbalanceOrg,
                name_dest=payload.nameDest,
                oldbalance_dest=payload.oldbalanceDest,
                raw_probability=raw_prob,
                calibrated_probability=calibrated_prob,
                risk_score=decision_res.risk_score,
                risk_tier=decision_res.risk_tier,
                decision=decision_res.decision,
                action_code=decision_res.operational_recommendation,
                decision_reasons=decision_res.reason_codes,
                latency_ms=latency_ms,
                model_version=model_ver,
                anomaly_score=anomaly_score,
                cold_start_applied=cold_start_applied,
                explanation_json=expl_json,
            )
        except Exception as e:
            logger.error(f"Audit log persistence failed for request {effective_req_id}: {e}")

    return FraudScoreResponse(
        request_id=effective_req_id,
        transaction_id=f"{payload.nameOrig}_{payload.step}",
        raw_probability=round(raw_prob, 6),
        calibrated_probability=round(calibrated_prob, 6),
        calibrated_fraud_probability=round(calibrated_prob, 6),
        risk_score=decision_res.risk_score,
        risk_tier=decision_res.risk_tier,
        decision=decision_res.decision,
        risk_decision=decision_res.decision,
        operational_recommendation=decision_res.operational_recommendation,
        action_code=decision_res.action_code,
        decision_reasons=decision_res.reason_codes,
        risk_weighted_investigation_priority=decision_res.risk_weighted_investigation_priority,
        triage_score=decision_res.risk_weighted_investigation_priority,
        anomaly_score=round(anomaly_score, 4) if anomaly_score is not None else None,
        latency_ms=latency_ms,
        model_version=model_ver,
        cold_start_applied=cold_start_applied,
        simulated_recommendation_only=True,
        recommendation_disclaimer=decision_res.recommendation_disclaimer,
        explanation=explanation_resp,
    )


@router.post("/score/batch", response_model=BatchScoreResponse, summary="Score Batch of Transactions")
async def score_batch_transactions(
    payload: BatchTransactionInput,
    request: Request,
) -> BatchScoreResponse:
    """Vectorized high-throughput batch transaction scoring up to configured batch limit."""
    state = request.app.state
    t0 = time.perf_counter()

    xgb_model = getattr(state, "xgb_model", None)
    calibrator = getattr(state, "calibrator", None)
    feature_builder = getattr(state, "feature_builder", None)
    policy_engine = getattr(state, "policy_engine", None)
    audit_repo = getattr(state, "audit_repo", None)
    explainer_service = getattr(state, "explainer_service", None)

    if xgb_model is None or calibrator is None or feature_builder is None or policy_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scoring engine models are not loaded into application state.",
        )

    transactions = payload.transactions
    n_tx = len(transactions)

    # Propagate batch-level settings if requested
    if payload.allow_cold_start:
        for tx in transactions:
            tx.allow_cold_start = True
    if payload.include_explanation:
        for tx in transactions:
            tx.include_explanation = True

    # 1. Vectorized feature construction
    X_batch, cold_flags = feature_builder.transform_batch(transactions)

    # 2. Vectorized model inference
    raw_probs = xgb_model.predict_proba(X_batch)[:, 1]

    # 3. Vectorized calibration
    calibrated_probs = calibrator.calibrate(raw_probs)

    # 4. Optional vectorized Autoencoder anomaly scoring
    anomaly_scores = [None] * n_tx
    ae_model = getattr(state, "ae_model", None)
    ae_scaler = getattr(state, "ae_scaler", None)
    if ae_model is not None and ae_scaler is not None:
        try:
            X_scaled = ae_scaler.transform(X_batch)
            if hasattr(ae_model, "compute_reconstruction_errors"):
                mse_errors, _ = ae_model.compute_reconstruction_errors(X_scaled)
            else:
                mse_errors, _ = ae_model.compute_reconstruction_error(X_scaled)
            anomaly_scores = [float(s) for s in ae_model.predict_anomaly_score(mse_errors)]
        except Exception as e:
            logger.debug(f"Batch autoencoder anomaly evaluation failed: {e}")

    # 5. Evaluate decisions and build results
    results: List[FraudScoreResponse] = []
    approvals = 0
    reviews = 0
    declines = 0
    total_amount = 0.0
    declined_amount = 0.0
    model_ver = getattr(state, "model_version", "1.0.0")

    for i, tx in enumerate(transactions):
        raw_p = float(raw_probs[i])
        cal_p = float(calibrated_probs[i])
        anom_s = anomaly_scores[i]
        c_flag = cold_flags[i]
        req_id = tx.get_effective_request_id()

        feats_dict, _ = feature_builder._build_features_dict(tx)
        decision_res = policy_engine.evaluate_decision(
            calibrated_prob=cal_p,
            amount=tx.amount,
            features_dict=feats_dict,
            anomaly_score=anom_s,
            cold_start_applied=c_flag,
        )

        total_amount += tx.amount
        if decision_res.decision == "APPROVE":
            approvals += 1
        elif decision_res.decision == "REVIEW":
            reviews += 1
        elif decision_res.decision == "DECLINE":
            declines += 1
            declined_amount += tx.amount

        # SHAP card if requested
        expl: Optional[FraudExplanationResponse] = None
        if tx.include_explanation and explainer_service is not None:
            try:
                expl = explainer_service.explain_transaction(
                    features_row=X_batch[i],
                    predicted_prob=cal_p,
                    decision=decision_res.decision,
                    anomaly_score=anom_s,
                )
            except Exception as e:
                logger.warning(f"Failed to generate SHAP explanation in batch item {i}: {e}")

        # Individual item response
        item_resp = FraudScoreResponse(
            request_id=req_id,
            transaction_id=f"{tx.nameOrig}_{tx.step}",
            raw_probability=round(raw_p, 6),
            calibrated_probability=round(cal_p, 6),
            calibrated_fraud_probability=round(cal_p, 6),
            risk_score=decision_res.risk_score,
            risk_tier=decision_res.risk_tier,
            decision=decision_res.decision,
            risk_decision=decision_res.decision,
            operational_recommendation=decision_res.operational_recommendation,
            action_code=decision_res.action_code,
            decision_reasons=decision_res.reason_codes,
            risk_weighted_investigation_priority=decision_res.risk_weighted_investigation_priority,
            triage_score=decision_res.risk_weighted_investigation_priority,
            anomaly_score=round(anom_s, 4) if anom_s is not None else None,
            latency_ms=0.0,  # Will be reflected in batch mean
            model_version=model_ver,
            cold_start_applied=c_flag,
            simulated_recommendation_only=True,
            recommendation_disclaimer=decision_res.recommendation_disclaimer,
            explanation=expl,
        )
        results.append(item_resp)

        # Asynchronously log decision
        if audit_repo is not None:
            expl_json = expl.model_dump_json() if expl else None
            try:
                await audit_repo.log_decision(
                    request_id=req_id,
                    step=tx.step,
                    corridor_type=tx.type,
                    amount=tx.amount,
                    name_orig=tx.nameOrig,
                    oldbalance_orig=tx.oldbalanceOrg,
                    name_dest=tx.nameDest,
                    oldbalance_dest=tx.oldbalanceDest,
                    raw_probability=raw_p,
                    calibrated_probability=cal_p,
                    risk_score=decision_res.risk_score,
                    risk_tier=decision_res.risk_tier,
                    decision=decision_res.decision,
                    action_code=decision_res.operational_recommendation,
                    decision_reasons=decision_res.reason_codes,
                    latency_ms=0.0,
                    model_version=model_ver,
                    anomaly_score=anom_s,
                    cold_start_applied=c_flag,
                    explanation_json=expl_json,
                )
            except Exception as e:
                logger.error(f"Audit log persistence failed for batch item {req_id}: {e}")

    total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    mean_latency_ms = round(total_latency_ms / max(n_tx, 1), 3)

    for item in results:
        item.latency_ms = mean_latency_ms

    summary = BatchScoreSummary(
        total_evaluated=n_tx,
        approvals=approvals,
        reviews=reviews,
        declines=declines,
        total_amount_evaluated=round(total_amount, 2),
        total_amount_declined=round(declined_amount, 2),
        mean_calibrated_probability=round(float(np.mean(calibrated_probs)), 6),
        mean_latency_ms=mean_latency_ms,
    )

    return BatchScoreResponse(
        batch_id=payload.get_effective_batch_id(),
        summary=summary,
        results=results,
        total_latency_ms=total_latency_ms,
    )
