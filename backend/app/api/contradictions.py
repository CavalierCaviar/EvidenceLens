"""Contradictions API: list claim pairs, contradiction explorer."""
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Analysis, Claim, ClaimPair, User
from app.schemas import ClaimOut, ClaimPairOut

router = APIRouter()


def _get_analysis(db: Session, analysis_id: str, user: User) -> Analysis:
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    from app.models import Collection
    col = db.get(Collection, analysis.collection_id)
    if not col or col.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return analysis


def _claim_to_out(claim: Optional[Claim]) -> Optional[ClaimOut]:
    if not claim:
        return None
    paper = claim.paper
    assessment = claim.assessment
    return ClaimOut(
        id=claim.id,
        paper_id=claim.paper_id,
        chunk_id=claim.chunk_id,
        page=claim.page,
        source_text=claim.source_text,
        claim_text=claim.claim_text,
        population=claim.population,
        intervention=claim.intervention,
        comparator=claim.comparator,
        outcome=claim.outcome,
        direction=claim.direction,
        effect=claim.effect,
        magnitude=claim.magnitude,
        claim_type=claim.claim_type,
        statistical_status=claim.statistical_status,
        effect_size=claim.effect_size,
        p_value=claim.p_value,
        confidence_interval=claim.confidence_interval,
        extraction_confidence=claim.extraction_confidence,
        paper_title=paper.title if paper else None,
        evidence_score=assessment.final_evidence_score if assessment else None,
        evidence_category=assessment.evidence_category if assessment else None,
    )


def _pair_to_out(db: Session, pair: ClaimPair) -> ClaimPairOut:
    claim_a = db.get(Claim, pair.claim_a_id)
    claim_b = db.get(Claim, pair.claim_b_id)

    # Determine confidence band
    if pair.relationship == "contradiction":
        if pair.p_contradiction >= 0.85:
            band = "high"
        elif pair.p_contradiction >= 0.65:
            band = "moderate"
        else:
            band = "low"
    else:
        conf = pair.relationship_confidence
        band = "high" if conf >= 0.75 else ("moderate" if conf >= 0.55 else "low")

    return ClaimPairOut(
        id=pair.id,
        claim_a_id=pair.claim_a_id,
        claim_b_id=pair.claim_b_id,
        semantic_similarity=pair.semantic_similarity,
        population_similarity=pair.population_similarity,
        intervention_similarity=pair.intervention_similarity,
        outcome_similarity=pair.outcome_similarity,
        nli_entailment_probability=pair.nli_entailment_probability,
        nli_contradiction_probability=pair.nli_contradiction_probability,
        nli_neutral_probability=pair.nli_neutral_probability,
        direction_conflict=pair.direction_conflict,
        statistical_conflict=pair.statistical_conflict,
        relationship=pair.relationship,
        p_contradiction=pair.p_contradiction,
        p_support=pair.p_support,
        p_neutral=pair.p_neutral,
        relationship_confidence=pair.relationship_confidence,
        contradiction_type=pair.contradiction_type,
        reason=pair.reason,
        confidence_band=band,
        claim_a=_claim_to_out(claim_a),
        claim_b=_claim_to_out(claim_b),
    )


@router.get("", response_model=list[ClaimPairOut])
def list_contradictions(
    analysis_id: str,
    relationship: Optional[str] = Query(default=None, description="Filter: contradiction|support|neutral"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """List claim pairs for an analysis, optionally filtered by relationship type."""
    _get_analysis(db, analysis_id, user)
    query = db.query(ClaimPair).filter(ClaimPair.analysis_id == analysis_id)

    if relationship:
        query = query.filter(ClaimPair.relationship == relationship)

    pairs = query.all()

    # Filter by confidence
    if min_confidence > 0:
        pairs = [p for p in pairs if p.relationship_confidence >= min_confidence]

    # Sort: contradictions first, then by confidence
    pairs.sort(key=lambda p: (p.relationship != "contradiction", -p.relationship_confidence))

    return [_pair_to_out(db, p) for p in pairs]


@router.get("/{pair_id}", response_model=ClaimPairOut)
def get_contradiction(
    analysis_id: str,
    pair_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_analysis(db, analysis_id, user)
    pair = db.get(ClaimPair, pair_id)
    if not pair or pair.analysis_id != analysis_id:
        raise HTTPException(status_code=404, detail="Claim pair not found")
    return _pair_to_out(db, pair)
