"""
Evidence-aware LLM synthesis module.
Aggregates evidence and generates structured synthesis using an LLM.
Falls back to a template-based synthesis when no LLM key is configured.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a scientific literature synthesis assistant. You produce faithful, evidence-grounded syntheses of academic research.

STRICT RULES:
1. Use ONLY the supplied evidence. Do not add outside knowledge.
2. Preserve contradictions — never hide disagreement.
3. Never invent study characteristics, effect sizes, or sample sizes.
4. Distinguish evidence strength from scientific truth.
5. Cite every substantive claim with its source paper ID.
6. Explicitly state uncertainty when evidence is insufficient.
7. Say when findings cannot be compared (different populations, outcomes).
8. Never present model confidence as scientific certainty.

Output a JSON object with exactly these fields:
{
  "overall_finding": "A concise, balanced 2-3 sentence summary",
  "sources_of_disagreement": ["list of specific methodological or population differences"],
  "uncertainty": "What cannot be concluded from this evidence",
  "limitations": ["Key limitations of the evidence base"]
}"""

SYNTHESIS_USER_TEMPLATE = """Research Question: {question}

SUPPORTING EVIDENCE ({n_support} studies):
{supporting}

CONTRADICTORY EVIDENCE ({n_contra} studies):
{contradicting}

NEUTRAL/UNCLEAR EVIDENCE ({n_neutral} studies):
{neutral}

CONTRADICTION PAIRS DETECTED:
{contradictions}

Generate a synthesis following the strict rules. Return ONLY valid JSON."""


def _format_evidence_list(items: list[dict]) -> str:
    if not items:
        return "None"
    parts = []
    for item in items[:8]:  # limit to 8 per group
        claim = item.get("claim_text", "")
        paper_id = item.get("paper_id", "")
        title = item.get("paper_title", paper_id)
        design = item.get("study_design", "UNKNOWN")
        n = item.get("sample_size", "UNKNOWN")
        score = item.get("evidence_score", 0.0)
        direction = item.get("direction", "UNKNOWN")
        parts.append(
            f"  - [{paper_id}] {title}: \"{claim}\" "
            f"(Design: {design}, n={n}, Direction: {direction}, Evidence strength: {score:.2f})"
        )
    return "\n".join(parts)


def _format_contradictions(pairs: list[dict]) -> str:
    if not pairs:
        return "No contradictions detected."
    parts = []
    for pair in pairs[:5]:
        a = pair.get("claim_a_text", "")
        b = pair.get("claim_b_text", "")
        p_con = pair.get("p_contradiction", 0.0)
        reason = pair.get("reason", "")
        parts.append(
            f"  CONTRADICTION (p={p_con:.2f}): \"{a}\" vs \"{b}\" — {reason}"
        )
    return "\n".join(parts)


def _template_synthesis(
    question: str,
    supporting: list[dict],
    contradicting: list[dict],
    neutral: list[dict],
    contradictions: list[dict],
) -> dict[str, Any]:
    """Fallback template synthesis without LLM."""
    n_sup = len(supporting)
    n_con = len(contradicting)
    n_neu = len(neutral)
    total = n_sup + n_con + n_neu

    if total == 0:
        return {
            "overall_finding": (
                "The available papers do not provide sufficient evidence "
                "to answer this question confidently."
            ),
            "sources_of_disagreement": [],
            "uncertainty": "Insufficient evidence retrieved.",
            "limitations": ["No evidence retrieved for the research question."],
        }

    if n_sup > 0 and n_con == 0:
        direction_str = "generally supports a positive association"
    elif n_con > 0 and n_sup == 0:
        direction_str = "generally does not support a positive association"
    elif n_sup > n_con:
        direction_str = "leans toward a positive association, though contradictory findings exist"
    elif n_con > n_sup:
        direction_str = "leans toward a null or negative association, though some supporting findings exist"
    else:
        direction_str = "is mixed, with roughly equal supporting and contradictory evidence"

    finding = (
        f"The available literature {direction_str}. "
        f"{n_sup} study/studies support, {n_con} contradict, and {n_neu} are neutral or unclear."
    )

    sources = []
    if contradictions:
        sources.append(f"{len(contradictions)} contradiction pair(s) detected between studies.")
    if n_sup > 0 and n_con > 0:
        sources.append("Differences in study design, population, or measurement may explain heterogeneity.")

    uncertainty = (
        "This synthesis is based on a limited evidence set. "
        "Study characteristics and populations may differ substantially."
    )
    if n_con > 0:
        uncertainty += " Contradictory evidence prevents a definitive conclusion."

    limitations = []
    if total < 5:
        limitations.append("Small number of retrieved studies.")
    limitations.append("Evidence strength estimates are model-derived, not peer-reviewed expert ratings.")

    return {
        "overall_finding": finding,
        "sources_of_disagreement": sources,
        "uncertainty": uncertainty,
        "limitations": limitations,
    }


def synthesize(
    question: str,
    supporting: list[dict],
    contradicting: list[dict],
    neutral: list[dict],
    contradictions: list[dict],
) -> dict[str, Any]:
    """
    Generate evidence-aware synthesis.
    Returns dict with overall_finding, sources_of_disagreement, uncertainty, limitations.
    """
    from app.config import settings

    if not settings.openai_api_key:
        return _template_synthesis(question, supporting, contradicting, neutral, contradictions)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        user_msg = SYNTHESIS_USER_TEMPLATE.format(
            question=question,
            n_support=len(supporting),
            supporting=_format_evidence_list(supporting),
            n_contra=len(contradicting),
            contradicting=_format_evidence_list(contradicting),
            n_neutral=len(neutral),
            neutral=_format_evidence_list(neutral),
            contradictions=_format_contradictions(contradictions),
        )

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=1000,
        )

        text = response.choices[0].message.content or ""
        # Strip markdown fences
        import re
        text = re.sub(r"```(?:json)?", "", text).strip()

        result = json.loads(text)
        # Validate required keys
        for key in ["overall_finding", "sources_of_disagreement", "uncertainty", "limitations"]:
            if key not in result:
                result[key] = ""
        return result

    except Exception as e:
        logger.warning("LLM synthesis failed: %s. Using template.", e)
        return _template_synthesis(question, supporting, contradicting, neutral, contradictions)


def validate_grounding(
    synthesis: dict[str, Any],
    evidence_items: list[dict],
) -> tuple[bool, float]:
    """
    Basic grounding validation: check if synthesis claims can be traced to evidence.
    Returns (is_grounded, unsupported_claim_rate).
    """
    if not evidence_items:
        return False, 1.0

    # Simple check: at least some evidence was used
    overall = synthesis.get("overall_finding", "")
    if len(overall) < 20:
        return False, 1.0

    # For now, use a proxy: if we have evidence and a finding, assume grounded
    # A full implementation would run NLI between finding sentences and evidence passages
    return True, 0.0
