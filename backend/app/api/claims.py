"""Claims API: list claims for an analysis with evidence explanations."""
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Analysis, Claim, EvidenceAssessment, Paper, User
from app.schemas import ClaimOut, EvidenceExplanationOut, MessageOut

router = APIRouter()


def _claim_to_out(claim: Claim) -> ClaimOut:
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


def _get_analysis(db: Session, analysis_id: str, user: User) -> Analysis:
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    from app.models import Collection
    col = db.get(Collection, analysis.collection_id)
    if not col or col.user_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return analysis


@router.get("", response_model=list[ClaimOut])
def list_claims(
    analysis_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_analysis(db, analysis_id, user)
    claims = db.query(Claim).filter(Claim.analysis_id == analysis_id).all()
    return [_claim_to_out(c) for c in claims]


@router.get("/{claim_id}", response_model=ClaimOut)
def get_claim(
    analysis_id: str,
    claim_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_analysis(db, analysis_id, user)
    claim = db.get(Claim, claim_id)
    if not claim or claim.analysis_id != analysis_id:
        raise HTTPException(status_code=404, detail="Claim not found")
    return _claim_to_out(claim)


@router.get("/{claim_id}/evidence", response_model=EvidenceExplanationOut)
def get_evidence_explanation(
    analysis_id: str,
    claim_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_analysis(db, analysis_id, user)
    claim = db.get(Claim, claim_id)
    if not claim or claim.analysis_id != analysis_id:
        raise HTTPException(status_code=404, detail="Claim not found")
    assessment = claim.assessment
    if not assessment:
        raise HTTPException(status_code=404, detail="No evidence assessment for this claim")

    factors = json.loads(assessment.contributing_factors_json) if assessment.contributing_factors_json else []
    limitations = json.loads(assessment.limitations_json) if assessment.limitations_json else []

    return EvidenceExplanationOut(
        assessment_id=assessment.id,
        claim_id=claim_id,
        final_evidence_score=assessment.final_evidence_score,
        evidence_category=assessment.evidence_category,
        rubric_score=assessment.rubric_score,
        linear_regression_score=assessment.linear_regression_score,
        study_design=assessment.study_design,
        sample_size=assessment.sample_size,
        precision_score=assessment.precision_score,
        directness_score=assessment.directness_score,
        risk_of_bias_score=assessment.risk_of_bias_score,
        statistical_support_score=assessment.statistical_support_score,
        consistency_score=assessment.consistency_score,
        contributing_factors=factors,
        limitations=limitations,
    )
