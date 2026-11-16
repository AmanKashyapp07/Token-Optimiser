from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from gateway.features.insights.models import get_length_predictor, predict_output_length
from gateway.tokenizer import TokenCounter


@dataclass
class TokenBudgetResult:
    """Telemetry and allocation result from ML-powered adaptive token budgeting."""
    allocated_budget: int
    predicted_tokens: float
    prediction_interval_margin: float
    confidence_level: float
    strategy: str  # 'ml_adaptive' or 'user_override'
    truncation_risk_warning: bool
    floor_applied: bool
    ceiling_applied: bool
    details: dict[str, Any]


class AdaptiveTokenBudgeter:
    """Operational ML Token Budgeter:
    Uses regularized Ridge Regression with statistical prediction intervals
    (hat_y + z * sigma) to allocate safe, dynamic output token ceilings.
    Prevents premature answer truncation while eliminating unconstrained token bloat.
    """

    def __init__(self, floor_tokens: int = 35, ceiling_tokens: int = 300):
        self.floor_tokens = floor_tokens
        self.ceiling_tokens = ceiling_tokens
        self.counter = TokenCounter()

    def allocate(
        self,
        prompt: str,
        language: str = "en",
        format_constraint: str | None = None,
        user_max_tokens: int | None = None,
        confidence_level: float = 0.95,
    ) -> TokenBudgetResult:
        input_tokens = self.counter.count(prompt)
        artifacts = get_length_predictor()

        # Run Ridge length prediction
        pred_res = predict_output_length(
            input_tokens=input_tokens,
            language=language,
            format_constraint=format_constraint,
            max_tokens=user_max_tokens,
        )
        predicted_tokens = float(pred_res["predicted_output_tokens"])

        # Statistical prediction interval margin:
        # z = 1.645 for 95% one-sided upper bound
        # sigma estimate from cross-validated MAE: sigma ~= 1.253 * MAE
        cv_mae = float(artifacts.cv_mae) if artifacts.cv_mae > 0 else 6.2
        sigma_est = 1.253 * cv_mae
        z_multiplier = 1.645 if confidence_level <= 0.95 else 1.96
        margin = math.ceil(z_multiplier * sigma_est)

        # Calculate safe adaptive upper bound: hat_y + margin
        adaptive_target = math.ceil(predicted_tokens + margin)

        floor_applied = False
        ceiling_applied = False

        if adaptive_target < self.floor_tokens:
            calculated_budget = self.floor_tokens
            floor_applied = True
        elif adaptive_target > self.ceiling_tokens:
            calculated_budget = self.ceiling_tokens
            ceiling_applied = True
        else:
            calculated_budget = adaptive_target

        truncation_risk = False
        strategy = "ml_adaptive"

        raw_ridge = float(pred_res.get("raw_linear_prediction", predicted_tokens))

        # If user explicitly requested a max_tokens limit
        if user_max_tokens is not None and user_max_tokens > 0:
            strategy = "user_override"
            if user_max_tokens < raw_ridge:
                # User's static limit is aggressively below expected minimum tokens
                truncation_risk = True
            allocated = min(user_max_tokens, calculated_budget)
        else:
            allocated = calculated_budget

        details = {
            "input_tokens": input_tokens,
            "language": language,
            "format_constraint": format_constraint or "none",
            "cv_mae": round(cv_mae, 2),
            "sigma_est": round(sigma_est, 2),
            "z_multiplier": z_multiplier,
            "raw_ridge_prediction": pred_res.get("raw_linear_prediction", predicted_tokens),
            "ridge_r2_score": artifacts.cv_r2,
        }

        return TokenBudgetResult(
            allocated_budget=allocated,
            predicted_tokens=predicted_tokens,
            prediction_interval_margin=margin,
            confidence_level=confidence_level,
            strategy=strategy,
            truncation_risk_warning=truncation_risk,
            floor_applied=floor_applied,
            ceiling_applied=ceiling_applied,
            details=details,
        )

    @staticmethod
    def static_heuristic_budget(input_tokens: int, factor: float = 1.5, floor: int = 40) -> int:
        """Baseline heuristic: proportional factor of input tokens."""
        return max(floor, math.ceil(input_tokens * factor))

    @staticmethod
    def static_uniform_budget(cap: int = 50) -> int:
        """Baseline static uniform capping."""
        return cap
