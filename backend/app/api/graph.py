"""Evidence graph API: return claim relationship graph for visualization."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Analysis, Claim, ClaimPair, Paper, User
from app.schemas import EvidenceGraphOut, GraphEdge, GraphNode

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


@router.get("", response_model=EvidenceGraphOut)
def get_evidence_graph(
    analysis_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Return the evidence relationship graph for an analysis.
    Nodes: research question, papers, claims.
    Edges: derived_from (claim->paper), supports/contradicts/neutral (claim-claim).
    """
    analysis = _get_analysis(db, analysis_id, user)
    claims = db.query(Claim).filter(Claim.analysis_id == analysis_id).all()
    pairs = db.query(ClaimPair).filter(ClaimPair.analysis_id == analysis_id).all()

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # Research question node
    nodes.append(GraphNode(
        id=f"q_{analysis_id}",
        type="question",
        label=analysis.research_question[:80],
        data={"research_question": analysis.research_question},
    ))

    # Paper nodes
    paper_ids_seen: set[str] = set()
    for claim in claims:
        pid = claim.paper_id
        if pid not in paper_ids_seen:
            paper = db.get(Paper, pid)
            paper_ids_seen.add(pid)
            nodes.append(GraphNode(
                id=f"paper_{pid}",
                type="paper",
                label=paper.title[:60] if paper else pid,
                data={
                    "paper_id": pid,
                    "title": paper.title if paper else "",
                    "year": paper.year if paper else "UNKNOWN",
                    "study_design": paper.study_design if paper else "UNKNOWN",
                    "sample_size": paper.sample_size if paper else "UNKNOWN",
                },
            ))

    # Claim nodes + derived_from edges
    for claim in claims:
        assessment = claim.assessment
        nodes.append(GraphNode(
            id=f"claim_{claim.id}",
            type="claim",
            label=claim.claim_text[:80],
            data={
                "claim_id": claim.id,
                "direction": claim.direction,
                "claim_type": claim.claim_type,
                "evidence_score": assessment.final_evidence_score if assessment else 0.0,
                "evidence_category": assessment.evidence_category if assessment else "Unknown",
            },
        ))
        edges.append(GraphEdge(
            source=f"claim_{claim.id}",
            target=f"paper_{claim.paper_id}",
            type="derived_from",
            weight=claim.extraction_confidence,
        ))

    # Claim pair edges
    for pair in pairs:
        rel = pair.relationship
        if rel == "support":
            edge_type = "supports"
            weight = pair.p_support
        elif rel == "contradiction":
            edge_type = "contradicts"
            weight = pair.p_contradiction
        else:
            edge_type = "neutral"
            weight = pair.p_neutral

        edges.append(GraphEdge(
            source=f"claim_{pair.claim_a_id}",
            target=f"claim_{pair.claim_b_id}",
            type=edge_type,
            weight=weight,
        ))

    return EvidenceGraphOut(nodes=nodes, edges=edges)
