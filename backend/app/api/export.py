"""Export API: machine-readable research export of analysis artifacts."""
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Analysis, Claim, ClaimPair, EvidenceAssessment, User

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


@router.get("/analyses/{analysis_id}")
def export_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Export complete analysis as machine-readable JSON.
    Contains: research_question, claims, claim_pairs, evidence_scores, synthesis, citations.
    """
    analysis = _get_analysis(db, analysis_id, user)

    claims = db.query(Claim).filter(Claim.analysis_id == analysis_id).all()
    pairs = db.query(ClaimPair).filter(ClaimPair.analysis_id == analysis_id).all()

    claims_export = []
    for c in claims:
        assessment = c.assessment
        claims_export.append({
            "claim_id": c.id,
            "paper_id": c.paper_id,
            "chunk_id": c.chunk_id,
            "page": c.page,
            "claim_text": c.claim_text,
            "source_text": c.source_text,
            "population": c.population,
            "intervention": c.intervention,
            "comparator": c.comparator,
            "outcome": c.outcome,
            "direction": c.direction,
            "effect": c.effect,
            "magnitude": c.magnitude,
            "claim_type": c.claim_type,
            "statistical_status": c.statistical_status,
            "effect_size": c.effect_size,
            "p_value": c.p_value,
            "confidence_interval": c.confidence_interval,
            "extraction_confidence": c.extraction_confidence,
            "evidence_score": assessment.final_evidence_score if assessment else None,
            "evidence_category": assessment.evidence_category if assessment else None,
            "rubric_score": assessment.rubric_score if assessment else None,
            "lr_score": assessment.linear_regression_score if assessment else None,
        })

    pairs_export = []
    for p in pairs:
        pairs_export.append({
            "pair_id": p.id,
            "claim_a_id": p.claim_a_id,
            "claim_b_id": p.claim_b_id,
            "relationship": p.relationship,
            "p_contradiction": p.p_contradiction,
            "p_support": p.p_support,
            "p_neutral": p.p_neutral,
            "relationship_confidence": p.relationship_confidence,
            "contradiction_type": p.contradiction_type,
            "semantic_similarity": p.semantic_similarity,
            "population_similarity": p.population_similarity,
            "intervention_similarity": p.intervention_similarity,
            "outcome_similarity": p.outcome_similarity,
            "nli_entailment": p.nli_entailment_probability,
            "nli_contradiction": p.nli_contradiction_probability,
            "nli_neutral": p.nli_neutral_probability,
            "direction_conflict": p.direction_conflict,
            "statistical_conflict": p.statistical_conflict,
            "reason": p.reason,
        })

    synthesis_data = {}
    if analysis.synthesis_json:
        try:
            synthesis_data = json.loads(analysis.synthesis_json)
        except Exception:
            pass

    export_doc = {
        "export_metadata": {
            "analysis_id": analysis_id,
            "research_question": analysis.research_question,
            "pipeline_mode": analysis.mode,
            "top_k": analysis.top_k,
            "status": analysis.status,
            "created_at": analysis.created_at.isoformat(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "claims": claims_export,
        "claim_pairs": pairs_export,
        "synthesis": synthesis_data,
        "uncertainty": analysis.uncertainty,
    }

    return JSONResponse(
        content=export_doc,
        headers={"Content-Disposition": f"attachment; filename=analysis_{analysis_id}.json"},
    )
