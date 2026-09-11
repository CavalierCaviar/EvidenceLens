"""
Evidence strength engine.
Implements:
  1. Transparent rubric (deterministic, interpretable)
  2. Linear Regression estimator (trained on expert annotations)
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

LR_MODEL_PATH = Path("./data/models/evidence_lr.pkl")
LR_SCALER_PATH = Path("./data/models/evidence_scaler.pkl")

STUDY_DESIGN_SCORES = {
    "Meta-analysis": 1.00,
    "Systematic Review": 0.90,
    "Randomized Controlled Trial": 0.85,
    "Cohort Study": 0.60,
    "Case-Control Study": 0.55,
    "Cross-Sectional Study": 0.45,
    "Case Study": 0.25,
    "Pilot Study": 0.30,
    "Other": 0.30,
    "Unknown": 0.20,
    "UNKNOWN": 0.20,
}

EVIDENCE_CATEGORIES = [
    (0.80, "Strong"),
    (0.60, "Moderate"),
    (0.40, "Weak"),
    (0.00, "Very Weak"),
]


@dataclass
class EvidenceFeatures:
    study_design: str
    sample_size: str  # raw string, may be UNKNOWN
    p_value: str = "UNKNOWN"
    confidence_interval: str = "UNKNOWN"
    direction: str = "UNKNOWN"
    statistical_status: str = "UNKNOWN"
    effect_size: str = "UNKNOWN"
    # Directness scores (0-1, 0.5 = unknown)
    population_match: float = 0.5
    intervention_match: float = 0.5
    outcome_match: float = 0.5


@dataclass
class EvidenceAssessmentResult:
    study_design_score: float
    log_sample_size: float
    precision_score: float
    directness_score: float
    risk_of_bias_score: float
    statistical_support_score: float
    consistency_score: float  # placeholder; set externally after aggregation
    rubric_score: float
    linear_regression_score: float
    final_evidence_score: float
    evidence_category: str
    contributing_factors: list[str]
    limitations: list[str]


def _parse_sample_size(ss_str: str) -> Optional[int]:
    if not ss_str or ss_str.upper() == "UNKNOWN":
        return None
    # Remove commas and take first number
    import re
    m = re.search(r"[\d,]+", ss_str)
    if m:
        return int(m.group().replace(",", ""))
    return None


def _log_sample_size_score(ss_str: str) -> float:
    """log(1 + n) normalized to [0, 1] assuming max useful n = 100000."""
    n = _parse_sample_size(ss_str)
    if n is None:
        return 0.3  # unknown penalty
    log_n = math.log1p(n)
    log_max = math.log1p(100_000)
    return min(1.0, log_n / log_max)


def _precision_score(p_value: str, ci: str) -> float:
    """Score based on p-value and confidence interval width."""
    score = 0.5  # default
    import re
    if p_value and p_value != "UNKNOWN":
        m = re.search(r"([0-9.e+\-]+)", p_value)
        if m:
            try:
                pv = float(m.group())
                if pv < 0.001:
                    score = 0.9
                elif pv < 0.01:
                    score = 0.75
                elif pv < 0.05:
                    score = 0.65
                elif pv < 0.10:
                    score = 0.45
                else:
                    score = 0.30
            except ValueError:
                pass
    # CI bonus (if CI is reported, precision is better known)
    if ci and ci != "UNKNOWN":
        score = min(1.0, score + 0.05)
    return score


def _statistical_support_score(statistical_status: str, direction: str, p_value: str) -> float:
    """Score based on whether results are statistically significant."""
    ss_lower = statistical_status.lower()
    if "significant" in ss_lower and "non" not in ss_lower:
        base = 0.80
    elif "non-significant" in ss_lower or statistical_status.upper() == "NULL":
        base = 0.30
    elif "not reported" in ss_lower:
        base = 0.40
    else:
        base = 0.50

    # Boost if p-value was also parsed successfully
    import re
    if p_value and p_value != "UNKNOWN":
        m = re.search(r"([0-9.e+\-]+)", p_value)
        if m:
            try:
                pv = float(m.group())
                if pv < 0.05:
                    base = min(1.0, base + 0.10)
            except ValueError:
                pass
    return base


def _risk_of_bias_score(study_design: str, sample_size: str) -> float:
    """
    Proxy for risk-of-bias. In a full implementation this would use explicit RoB checklists.
    RCTs and meta-analyses get higher base scores; small studies are penalized.
    """
    design_base = STUDY_DESIGN_SCORES.get(study_design, 0.30)
    n = _parse_sample_size(sample_size)
    size_penalty = 0.0
    if n is not None and n < 30:
        size_penalty = 0.15
    elif n is not None and n < 100:
        size_penalty = 0.08
    return max(0.0, design_base - size_penalty)


def build_feature_vector(ef: EvidenceFeatures) -> np.ndarray:
    """Build the feature vector for Linear Regression evidence estimation."""
    design_score = STUDY_DESIGN_SCORES.get(ef.study_design, 0.20)
    log_ss = _log_sample_size_score(ef.sample_size)
    precision = _precision_score(ef.p_value, ef.confidence_interval)
    directness = (ef.population_match + ef.intervention_match + ef.outcome_match) / 3.0
    rob = _risk_of_bias_score(ef.study_design, ef.sample_size)
    stat_support = _statistical_support_score(ef.statistical_status, ef.direction, ef.p_value)

    return np.array([
        design_score,
        log_ss,
        precision,
        directness,
        rob,
        stat_support,
    ], dtype=np.float32)


RUBRIC_WEIGHTS = {
    "study_design": 0.25,
    "sample_size": 0.20,
    "precision": 0.15,
    "directness": 0.15,
    "risk_of_bias": 0.15,
    "statistical_support": 0.10,
}


def compute_rubric_score(ef: EvidenceFeatures) -> tuple[float, list[str], list[str]]:
    """
    Compute evidence score using transparent weighted rubric.
    Returns (score, contributing_factors, limitations).
    """
    design_score = STUDY_DESIGN_SCORES.get(ef.study_design, 0.20)
    log_ss = _log_sample_size_score(ef.sample_size)
    precision = _precision_score(ef.p_value, ef.confidence_interval)
    directness = (ef.population_match + ef.intervention_match + ef.outcome_match) / 3.0
    rob = _risk_of_bias_score(ef.study_design, ef.sample_size)
    stat_support = _statistical_support_score(ef.statistical_status, ef.direction, ef.p_value)

    score = (
        RUBRIC_WEIGHTS["study_design"] * design_score
        + RUBRIC_WEIGHTS["sample_size"] * log_ss
        + RUBRIC_WEIGHTS["precision"] * precision
        + RUBRIC_WEIGHTS["directness"] * directness
        + RUBRIC_WEIGHTS["risk_of_bias"] * rob
        + RUBRIC_WEIGHTS["statistical_support"] * stat_support
    )

    contributing_factors = []
    limitations = []

    if design_score >= 0.80:
        contributing_factors.append(f"High-quality study design ({ef.study_design})")
    elif design_score <= 0.35:
        limitations.append(f"Lower-quality study design ({ef.study_design})")

    n = _parse_sample_size(ef.sample_size)
    if n is not None:
        if n >= 500:
            contributing_factors.append(f"Large sample size (n={n:,})")
        elif n < 50:
            limitations.append(f"Small sample size (n={n:,})")
    else:
        limitations.append("Sample size unknown")

    if precision >= 0.75:
        contributing_factors.append("Strong statistical precision (p < 0.01)")
    elif precision <= 0.40:
        limitations.append("Weak or unreported statistical precision")

    if directness >= 0.70:
        contributing_factors.append("Study closely matches research question")
    elif directness <= 0.35:
        limitations.append("Study may not directly address the research question")

    if stat_support >= 0.75:
        contributing_factors.append("Statistically significant result reported")
    elif stat_support <= 0.35:
        limitations.append("Non-significant or unreported statistical result")

    return float(score), contributing_factors, limitations


def _load_lr_model() -> tuple[Optional[Ridge], Optional[StandardScaler]]:
    if LR_MODEL_PATH.exists() and LR_SCALER_PATH.exists():
        try:
            with open(LR_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(LR_SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e:
            logger.warning("Failed to load LR evidence model: %s", e)
    return None, None


def _score_to_category(score: float) -> str:
    for threshold, label in EVIDENCE_CATEGORIES:
        if score >= threshold:
            return label
    return "Very Weak"


def assess_evidence(
    ef: EvidenceFeatures,
    consistency_score: float = 0.5,
) -> EvidenceAssessmentResult:
    """Compute evidence strength using rubric and optional LR model."""
    feature_vec = build_feature_vector(ef)
    rubric_score, factors, limitations = compute_rubric_score(ef)

    # Try LR model
    lr_model, lr_scaler = _load_lr_model()
    if lr_model is not None and lr_scaler is not None:
        try:
            x = np.append(feature_vec, consistency_score).reshape(1, -1)
            x_scaled = lr_scaler.transform(x)
            lr_score = float(np.clip(lr_model.predict(x_scaled)[0], 0.0, 1.0))
        except Exception:
            lr_score = rubric_score
    else:
        lr_score = rubric_score

    # Final score: average of rubric and LR if LR is available
    if lr_model is not None:
        final_score = 0.5 * rubric_score + 0.5 * lr_score
    else:
        final_score = rubric_score

    return EvidenceAssessmentResult(
        study_design_score=STUDY_DESIGN_SCORES.get(ef.study_design, 0.20),
        log_sample_size=_log_sample_size_score(ef.sample_size),
        precision_score=_precision_score(ef.p_value, ef.confidence_interval),
        directness_score=(ef.population_match + ef.intervention_match + ef.outcome_match) / 3.0,
        risk_of_bias_score=_risk_of_bias_score(ef.study_design, ef.sample_size),
        statistical_support_score=_statistical_support_score(ef.statistical_status, ef.direction, ef.p_value),
        consistency_score=consistency_score,
        rubric_score=rubric_score,
        linear_regression_score=lr_score,
        final_evidence_score=final_score,
        evidence_category=_score_to_category(final_score),
        contributing_factors=factors,
        limitations=limitations,
    )


def train_lr_model(
    features: np.ndarray,  # (N, 7) — 6 base features + consistency
    targets: np.ndarray,   # (N,) expert scores in [0, 1]
    save: bool = True,
) -> tuple[Ridge, StandardScaler]:
    """Train the Linear Regression evidence strength estimator."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_scaled, targets)

    if save:
        LR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LR_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(LR_SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    logger.info(
        "Evidence LR model trained. R2=%.3f",
        model.score(X_scaled, targets),
    )
    return model, scaler
