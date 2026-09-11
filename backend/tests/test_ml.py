import pytest
import numpy as np

from app.ml.claim_extractor import _heuristic_extract
from app.ml.contradiction_classifier import _direction_conflict, _statistical_conflict
from app.ml.evidence_engine import compute_rubric_score, _score_to_category, _log_sample_size_score
from app.ml.embedder import encode, cosine_sim

def test_heuristic_extract():
    text = "We conclude that X increases Y. However, Z decreases Y."
    claims = _heuristic_extract(text)
    assert isinstance(claims, list)

def test_direction_conflict():
    conflict = _direction_conflict("increases", "decreases")
    assert conflict == 1

def test_statistical_conflict():
    conflict = _statistical_conflict("p < 0.05", "no significant difference")
    assert isinstance(conflict, (int, float, bool))

def test_compute_rubric_score():
    score = compute_rubric_score(sample_size=1000, study_design="Randomized Controlled Trial", p_value=0.01)
    assert 0 <= score <= 1

def test_score_to_category():
    cat1 = _score_to_category(0.9)
    assert cat1 == "Strong"
    cat2 = _score_to_category(0.2)
    assert cat2 == "Very Weak"

def test_log_sample_size_score():
    score_10 = _log_sample_size_score(10)
    score_1000 = _log_sample_size_score(1000)
    assert score_1000 > score_10

def test_encode():
    text = "Sample claim text"
    embedding = encode(text)
    assert hasattr(embedding, "shape") or isinstance(embedding, (list, tuple))

def test_cosine_sim():
    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    assert cosine_sim(v1, v2) == pytest.approx(1.0)
    assert cosine_sim(v1, v3) == pytest.approx(0.0)
