"""NLI model wrapper for scientific claim pair classification."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_using_heuristic = False


def _load_model():
    global _model, _tokenizer, _using_heuristic
    if _model is not None or _using_heuristic:
        return
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        from app.config import settings

        model_name = settings.nli_model
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()
        logger.info("Loaded NLI model: %s", model_name)
    except Exception as e:
        logger.warning("NLI model unavailable (%s). Using heuristic fallback.", e)
        _using_heuristic = True


@dataclass
class NLIResult:
    entailment: float
    contradiction: float
    neutral: float

    @property
    def label(self) -> str:
        scores = {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
        }
        return max(scores, key=scores.get)  # type: ignore


def _heuristic_nli(premise: str, hypothesis: str) -> NLIResult:
    """
    Simple keyword heuristic NLI fallback.
    Uses negation and direction keywords to estimate probabilities.
    """
    neg_words = {"not", "no", "never", "neither", "without", "failed", "fail",
                 "negative", "decreased", "reduced", "worsened", "null", "absence"}
    pos_words = {"improved", "increased", "enhanced", "positive", "significant",
                 "beneficial", "effective", "reduces", "better", "higher"}

    def polarity(text: str) -> float:
        words = set(text.lower().split())
        pos_count = len(words & pos_words)
        neg_count = len(words & neg_words)
        return pos_count - neg_count

    p_pol = polarity(premise)
    h_pol = polarity(hypothesis)

    # Same polarity → likely entailment/support
    # Opposite polarity → likely contradiction
    diff = p_pol * h_pol
    if diff > 0:
        return NLIResult(entailment=0.65, contradiction=0.15, neutral=0.20)
    elif diff < 0:
        return NLIResult(entailment=0.10, contradiction=0.70, neutral=0.20)
    else:
        return NLIResult(entailment=0.25, contradiction=0.25, neutral=0.50)


def run_nli(premise: str, hypothesis: str) -> NLIResult:
    """Run NLI on a premise-hypothesis pair."""
    _load_model()
    if _using_heuristic:
        return _heuristic_nli(premise, hypothesis)

    try:
        import torch
        inputs = _tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = _model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        # Map model labels: cross-encoder/nli-deberta-v3-small uses [contradiction, entailment, neutral]
        # Label order may vary — check model config
        label_names = _model.config.id2label
        label_map = {v.lower(): i for i, v in label_names.items()}

        def get_prob(name: str) -> float:
            for key, idx in label_map.items():
                if name in key:
                    return probs[idx] if isinstance(probs, list) else probs
            return 1 / 3

        return NLIResult(
            entailment=get_prob("entailment"),
            contradiction=get_prob("contradiction"),
            neutral=get_prob("neutral"),
        )
    except Exception as e:
        logger.warning("NLI inference error: %s. Using heuristic.", e)
        return _heuristic_nli(premise, hypothesis)


def batch_nli(pairs: list[tuple[str, str]]) -> list[NLIResult]:
    """Run NLI on multiple pairs."""
    return [run_nli(p, h) for p, h in pairs]
