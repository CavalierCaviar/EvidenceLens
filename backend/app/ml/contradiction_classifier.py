"""
Contradiction classifier.
Logistic Regression over NLI + structured PICO features.
Provides calibrated probability estimates for SUPPORT / CONTRADICTION / NEUTRAL.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from app.ml.nli import NLIResult, run_nli
from app.ml.embedder import cosine_sim, encode

logger = logging.getLogger(__name__)

LABEL_MAP = {"support": 0, "contradiction": 1, "neutral": 2}
LABEL_NAMES = {0: "support", 1: "contradiction", 2: "neutral"}

CONTRADICTION_HIGH = 0.85
CONTRADICTION_MODERATE = 0.65

MODEL_PATH = Path("./data/models/contradiction_classifier.pkl")
SCALER_PATH = Path("./data/models/contradiction_scaler.pkl")


@dataclass
class ContradictionResult:
    relationship: str  # support | contradiction | neutral
    p_contradiction: float
    p_support: float
    p_neutral: float
    relationship_confidence: float
    contradiction_type: str  # direct | directional | statistical | magnitude | none
    reason: str
    confidence_band: str  # high | moderate | low


def _direction_conflict(dir_a: str, dir_b: str) -> int:
    """1 if directions conflict, 0 otherwise."""
    pos = {"positive", "increase", "improve", "higher", "greater", "beneficial"}
    neg = {"negative", "decrease", "reduce", "lower", "null", "no effect", "worsened"}
    a_pos = dir_a.lower() in pos
    a_neg = dir_a.lower() in neg
    b_pos = dir_b.lower() in pos
    b_neg = dir_b.lower() in neg
    if (a_pos and b_neg) or (a_neg and b_pos):
        return 1
    return 0


def _statistical_conflict(stat_a: str, stat_b: str) -> int:
    """1 if one claim is significant and the other is not."""
    sig_a = "significant" in stat_a.lower() and "non" not in stat_a.lower()
    sig_b = "significant" in stat_b.lower() and "non" not in stat_b.lower()
    nonsig_a = "non-significant" in stat_a.lower() or stat_a.upper() == "NULL"
    nonsig_b = "non-significant" in stat_b.lower() or stat_b.upper() == "NULL"
    if (sig_a and nonsig_b) or (sig_b and nonsig_a):
        return 1
    return 0


def _text_sim(text_a: str, text_b: str) -> float:
    """Cosine similarity between two text embeddings."""
    try:
        embs = encode([text_a, text_b])
        return float(cosine_sim(embs[0], embs[1]))
    except Exception:
        return 0.5


def _field_sim(a: Optional[str], b: Optional[str]) -> float:
    """Soft similarity between two string fields."""
    if not a or not b or a == "UNKNOWN" or b == "UNKNOWN":
        return 0.5  # unknown, don't penalize
    if a.lower() == b.lower():
        return 1.0
    # Check for partial overlap
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    intersection = words_a & words_b
    union = words_a | words_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _build_features(
    claim_a_text: str,
    claim_b_text: str,
    nli: NLIResult,
    dir_a: str = "UNKNOWN",
    dir_b: str = "UNKNOWN",
    stat_a: str = "UNKNOWN",
    stat_b: str = "UNKNOWN",
    pop_a: str = "UNKNOWN",
    pop_b: str = "UNKNOWN",
    intervention_a: str = "UNKNOWN",
    intervention_b: str = "UNKNOWN",
    outcome_a: str = "UNKNOWN",
    outcome_b: str = "UNKNOWN",
) -> np.ndarray:
    """Build feature vector for contradiction classification."""
    sem_sim = _text_sim(claim_a_text, claim_b_text)
    dir_conf = _direction_conflict(dir_a, dir_b)
    stat_conf = _statistical_conflict(stat_a, stat_b)
    pop_sim = _field_sim(pop_a, pop_b)
    int_sim = _field_sim(intervention_a, intervention_b)
    out_sim = _field_sim(outcome_a, outcome_b)

    features = np.array([
        sem_sim,
        nli.entailment,
        nli.contradiction,
        nli.neutral,
        float(dir_conf),
        float(stat_conf),
        pop_sim,
        int_sim,
        out_sim,
        # Interaction features
        nli.contradiction * float(dir_conf),
        nli.entailment * (1.0 - float(dir_conf)),
        (pop_sim + int_sim + out_sim) / 3.0,  # structural similarity
    ], dtype=np.float32)
    return features


def _get_contradiction_type(
    nli: NLIResult,
    dir_conflict: int,
    stat_conflict: int,
    dir_a: str,
    dir_b: str,
) -> str:
    if dir_conflict and "positive" in (dir_a, dir_b) and "negative" in (dir_a, dir_b):
        return "directional"
    if stat_conflict:
        return "statistical"
    if dir_conflict:
        return "directional"
    if nli.contradiction > 0.6:
        return "direct"
    return "none"


def _get_reason(
    ct_type: str,
    nli: NLIResult,
    dir_a: str,
    dir_b: str,
    stat_a: str,
    stat_b: str,
    outcome_a: str,
    outcome_b: str,
    pop_sim: float,
    int_sim: float,
    out_sim: float,
) -> str:
    parts = []
    if ct_type == "directional":
        parts.append(f"Direction conflict: {dir_a} vs {dir_b}.")
    if ct_type == "statistical":
        parts.append(f"Statistical conflict: {stat_a} vs {stat_b}.")
    if nli.contradiction > 0.6:
        parts.append(f"NLI contradiction score: {nli.contradiction:.2f}.")
    if pop_sim < 0.3:
        parts.append("Different populations may explain the discrepancy.")
    if int_sim < 0.3:
        parts.append("Different interventions.")
    if out_sim < 0.3:
        parts.append("Outcomes may not be directly comparable.")
    if not parts:
        if nli.entailment > 0.5:
            parts.append("Claims appear to support each other.")
        else:
            parts.append("Claims are not directly comparable.")
    return " ".join(parts)


def _build_default_model() -> tuple[LogisticRegression, StandardScaler]:
    """Build a default logistic regression with prior-based initialization."""
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    scaler = StandardScaler()
    return model, scaler


def _load_or_init_model() -> tuple[Optional[LogisticRegression], Optional[StandardScaler]]:
    """Load trained model if it exists."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e:
            logger.warning("Failed to load classifier: %s", e)
    return None, None


def _nli_only_probabilities(nli: NLIResult, dir_conf: int, stat_conf: int) -> tuple[float, float, float]:
    """
    Derive class probabilities from NLI + structured features without a trained LR.
    p_support, p_contradiction, p_neutral
    """
    # Base from NLI
    p_sup = nli.entailment
    p_con = nli.contradiction
    p_neu = nli.neutral

    # Boost contradiction if structural conflict
    if dir_conf or stat_conf:
        p_con = min(1.0, p_con + 0.25)
        p_sup = max(0.0, p_sup - 0.15)

    # Normalize
    total = p_sup + p_con + p_neu
    if total > 0:
        p_sup /= total
        p_con /= total
        p_neu /= total

    return p_sup, p_con, p_neu


def classify(
    claim_a_text: str,
    claim_b_text: str,
    dir_a: str = "UNKNOWN",
    dir_b: str = "UNKNOWN",
    stat_a: str = "UNKNOWN",
    stat_b: str = "UNKNOWN",
    pop_a: str = "UNKNOWN",
    pop_b: str = "UNKNOWN",
    intervention_a: str = "UNKNOWN",
    intervention_b: str = "UNKNOWN",
    outcome_a: str = "UNKNOWN",
    outcome_b: str = "UNKNOWN",
) -> ContradictionResult:
    """Classify the relationship between two claims."""
    # Run NLI
    nli = run_nli(claim_a_text, claim_b_text)

    dir_conf = _direction_conflict(dir_a, dir_b)
    stat_conf = _statistical_conflict(stat_a, stat_b)
    pop_sim = _field_sim(pop_a, pop_b)
    int_sim = _field_sim(intervention_a, intervention_b)
    out_sim = _field_sim(outcome_a, outcome_b)

    # Try trained model
    model, scaler = _load_or_init_model()
    if model is not None and scaler is not None:
        try:
            features = _build_features(
                claim_a_text, claim_b_text, nli,
                dir_a, dir_b, stat_a, stat_b,
                pop_a, pop_b, intervention_a, intervention_b, outcome_a, outcome_b,
            )
            features_scaled = scaler.transform(features.reshape(1, -1))
            probs = model.predict_proba(features_scaled)[0]
            # Map to named labels
            classes = model.classes_
            p_sup = float(probs[list(classes).index(0)] if 0 in classes else 0.33)
            p_con = float(probs[list(classes).index(1)] if 1 in classes else 0.33)
            p_neu = float(probs[list(classes).index(2)] if 2 in classes else 0.34)
        except Exception:
            p_sup, p_con, p_neu = _nli_only_probabilities(nli, dir_conf, stat_conf)
    else:
        p_sup, p_con, p_neu = _nli_only_probabilities(nli, dir_conf, stat_conf)

    # Determine label
    scores = {"support": p_sup, "contradiction": p_con, "neutral": p_neu}
    relationship = max(scores, key=scores.get)  # type: ignore
    confidence = scores[relationship]

    # Confidence band
    if relationship == "contradiction":
        if p_con >= CONTRADICTION_HIGH:
            band = "high"
        elif p_con >= CONTRADICTION_MODERATE:
            band = "moderate"
        else:
            band = "low"
    else:
        band = "high" if confidence >= 0.75 else ("moderate" if confidence >= 0.55 else "low")

    ct_type = _get_contradiction_type(nli, dir_conf, stat_conf, dir_a, dir_b)
    reason = _get_reason(ct_type, nli, dir_a, dir_b, stat_a, stat_b, outcome_a, outcome_b, pop_sim, int_sim, out_sim)

    return ContradictionResult(
        relationship=relationship,
        p_contradiction=p_con,
        p_support=p_sup,
        p_neutral=p_neu,
        relationship_confidence=confidence,
        contradiction_type=ct_type,
        reason=reason,
        confidence_band=band,
    )


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    save: bool = True,
) -> tuple[LogisticRegression, StandardScaler]:
    """Train the logistic regression classifier on labelled data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_scaled, labels)

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    return model, scaler
