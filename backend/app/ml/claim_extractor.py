"""
Claim extractor and normalizer.
Uses an LLM (OpenAI-compatible) to extract PICO-structured claims from passages.
Falls back to a pattern-based heuristic when no LLM key is configured.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

CLAIM_TYPES = [
    "RESULT", "ASSOCIATION", "CAUSAL CLAIM",
    "NULL FINDING", "METHODOLOGICAL CLAIM", "BACKGROUND CLAIM",
]

DIRECTIONS = ["positive", "negative", "null", "unclear", "UNKNOWN"]


@dataclass
class ExtractedClaim:
    claim_text: str
    source_text: str
    population: str = "UNKNOWN"
    intervention: str = "UNKNOWN"
    comparator: str = "UNKNOWN"
    outcome: str = "UNKNOWN"
    direction: str = "UNKNOWN"
    effect: str = "UNKNOWN"
    magnitude: str = "UNKNOWN"
    claim_type: str = "RESULT"
    statistical_status: str = "UNKNOWN"
    effect_size: str = "UNKNOWN"
    p_value: str = "UNKNOWN"
    confidence_interval: str = "UNKNOWN"
    extraction_confidence: float = 0.5


CLAIM_EXTRACTION_PROMPT = """You are a scientific claim extractor. Given a passage from an academic paper, extract all research claims.

For each claim, return a JSON object with these exact fields (use "UNKNOWN" for any field you cannot determine from the text - never invent values):
{
  "claim_text": "A concise normalized claim statement",
  "population": "Who was studied",
  "intervention": "What intervention/exposure was applied",
  "comparator": "Comparison condition (or UNKNOWN)",
  "outcome": "What outcome was measured",
  "direction": "positive/negative/null/unclear/UNKNOWN",
  "effect": "Description of the effect",
  "magnitude": "Quantitative magnitude or UNKNOWN",
  "claim_type": "RESULT/ASSOCIATION/CAUSAL CLAIM/NULL FINDING/METHODOLOGICAL CLAIM/BACKGROUND CLAIM",
  "statistical_status": "significant/non-significant/not reported/UNKNOWN",
  "effect_size": "Effect size value or UNKNOWN",
  "p_value": "p-value or UNKNOWN",
  "confidence_interval": "CI or UNKNOWN",
  "extraction_confidence": 0.0-1.0
}

Return a JSON array of claim objects. Extract only meaningful research claims, not background statements.

PASSAGE:
{passage}

Return ONLY valid JSON array, no other text."""


def _parse_llm_response(response_text: str, source_text: str) -> list[ExtractedClaim]:
    """Parse LLM JSON response into ExtractedClaim objects."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", response_text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        claims = []
        for item in data:
            if not isinstance(item, dict):
                continue
            claim = ExtractedClaim(
                claim_text=item.get("claim_text", "").strip(),
                source_text=source_text,
                population=item.get("population", "UNKNOWN") or "UNKNOWN",
                intervention=item.get("intervention", "UNKNOWN") or "UNKNOWN",
                comparator=item.get("comparator", "UNKNOWN") or "UNKNOWN",
                outcome=item.get("outcome", "UNKNOWN") or "UNKNOWN",
                direction=item.get("direction", "UNKNOWN") or "UNKNOWN",
                effect=item.get("effect", "UNKNOWN") or "UNKNOWN",
                magnitude=item.get("magnitude", "UNKNOWN") or "UNKNOWN",
                claim_type=item.get("claim_type", "RESULT") or "RESULT",
                statistical_status=item.get("statistical_status", "UNKNOWN") or "UNKNOWN",
                effect_size=item.get("effect_size", "UNKNOWN") or "UNKNOWN",
                p_value=item.get("p_value", "UNKNOWN") or "UNKNOWN",
                confidence_interval=item.get("confidence_interval", "UNKNOWN") or "UNKNOWN",
                extraction_confidence=float(item.get("extraction_confidence", 0.5)),
            )
            if claim.claim_text:
                claims.append(claim)
        return claims
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse LLM claim response: %s", e)
        return []


def _heuristic_extract(passage: str) -> list[ExtractedClaim]:
    """
    Simple heuristic claim extraction when LLM is unavailable.
    Detects result sentences and infers direction from keywords.
    """
    result_patterns = [
        r"significant[ly]?\s+\w+(?:ed|es|ing)?",
        r"(?:increase|decrease|improve|reduce|enhance|worsen)[sd]?\s+\w+",
        r"no\s+significant\s+\w+",
        r"associated\s+with",
        r"effect\s+of\s+\w+\s+on",
        r"(?:higher|lower|greater|less)\s+\w+",
    ]

    neg_words = {"not", "no", "never", "failed", "negative", "decreased",
                 "reduced", "null", "absence", "without"}
    pos_words = {"improved", "increased", "enhanced", "positive", "significant",
                 "beneficial", "effective", "higher", "greater"}

    claims = []
    sentences = re.split(r'(?<=[.!?])\s+', passage)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30:
            continue
        if any(re.search(pat, sent, re.IGNORECASE) for pat in result_patterns):
            words = set(sent.lower().split())
            pos = len(words & pos_words)
            neg = len(words & neg_words)
            if neg > pos:
                direction = "negative" if "decrease" in sent.lower() else "null"
            elif pos > 0:
                direction = "positive"
            else:
                direction = "unclear"

            # Statistical status
            if re.search(r'p\s*[<=>]\s*0\.\d+|significant', sent, re.I):
                stat_status = "significant" if "significant" in sent.lower() else "reported"
            elif re.search(r'no\s+significant', sent, re.I):
                stat_status = "non-significant"
            else:
                stat_status = "UNKNOWN"

            claims.append(ExtractedClaim(
                claim_text=sent[:500],
                source_text=passage[:500],
                direction=direction,
                statistical_status=stat_status,
                extraction_confidence=0.4,
            ))

    return claims


def extract_claims(
    passage: str,
    max_claims: int = 5,
) -> list[ExtractedClaim]:
    """
    Extract research claims from a passage.
    Uses LLM if configured, otherwise uses heuristic.
    """
    from app.config import settings

    if not settings.openai_api_key:
        return _heuristic_extract(passage)[:max_claims]

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        prompt = CLAIM_EXTRACTION_PROMPT.format(passage=passage[:3000])
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        text = response.choices[0].message.content or ""
        claims = _parse_llm_response(text, passage)
        return claims[:max_claims]
    except Exception as e:
        logger.warning("LLM claim extraction failed: %s. Using heuristic.", e)
        return _heuristic_extract(passage)[:max_claims]


def extract_claims_batch(
    passages: list[tuple[str, str]],  # (chunk_text, chunk_id)
    max_per_chunk: int = 3,
) -> list[tuple[str, list[ExtractedClaim]]]:
    """Extract claims from multiple passages. Returns list of (chunk_id, claims)."""
    results = []
    for text, chunk_id in passages:
        claims = extract_claims(text, max_claims=max_per_chunk)
        results.append((chunk_id, claims))
    return results
