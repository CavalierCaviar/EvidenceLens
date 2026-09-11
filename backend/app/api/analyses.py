"""Analyses endpoints: run analysis, get results, list analyses."""
import json
import logging
import threading

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.deps import get_current_user
from app.models import Analysis, Collection, User
from app.schemas import AnalysisOut, MessageOut, QueryRequest, SynthesisOut

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_collection(db: Session, collection_id: str, user: User) -> Collection:
    col = db.get(Collection, collection_id)
    if not col or col.user_id != user.id:
        raise HTTPException(status_code=404, detail="Collection not found")
    return col


def _run_in_thread(analysis_id: str):
    """Run analysis pipeline in a background thread with its own DB session."""
    db = SessionLocal()
    try:
        analysis = db.get(Analysis, analysis_id)
        if not analysis:
            return
        from app.services.analysis_service import run_analysis
        run_analysis(db, analysis)
    except Exception as e:
        logger.error("Background analysis %s failed: %s", analysis_id, e)
        db2 = SessionLocal()
        try:
            a = db2.get(Analysis, analysis_id)
            if a:
                a.status = "failed"
                a.error_message = str(e)[:500]
                db2.commit()
        finally:
            db2.close()
    finally:
        db.close()


def _analysis_to_out(db: Session, analysis: Analysis) -> AnalysisOut:
    synthesis = None
    if analysis.status == "complete" and analysis.synthesis_json:
        try:
            from app.services.analysis_service import build_synthesis_out
            synthesis = build_synthesis_out(db, analysis)
        except Exception as e:
            logger.warning("Failed to build synthesis out: %s", e)

    return AnalysisOut(
        id=analysis.id,
        collection_id=analysis.collection_id,
        research_question=analysis.research_question,
        status=analysis.status,
        mode=analysis.mode,
        top_k=analysis.top_k,
        created_at=analysis.created_at,
        uncertainty=analysis.uncertainty,
        error_message=analysis.error_message,
        synthesis=synthesis,
    )


@router.post("", response_model=AnalysisOut, status_code=201)
def run_analysis_endpoint(
    collection_id: str,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Start a new analysis for a collection."""
    _check_collection(db, collection_id, user)

    analysis = Analysis(
        collection_id=collection_id,
        user_id=user.id,
        research_question=body.research_question,
        mode=body.mode,
        top_k=body.top_k,
        status="running",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Run in background thread (FastAPI BackgroundTasks + separate DB session)
    background_tasks.add_task(_run_in_thread, analysis.id)

    return _analysis_to_out(db, analysis)


@router.get("", response_model=list[AnalysisOut])
def list_analyses(
    collection_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_collection(db, collection_id, user)
    analyses = (
        db.query(Analysis)
        .filter(Analysis.collection_id == collection_id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    return [_analysis_to_out(db, a) for a in analyses]


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_analysis(
    collection_id: str,
    analysis_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_collection(db, collection_id, user)
    analysis = db.get(Analysis, analysis_id)
    if not analysis or analysis.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _analysis_to_out(db, analysis)


@router.delete("/{analysis_id}", response_model=MessageOut)
def delete_analysis(
    collection_id: str,
    analysis_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_collection(db, collection_id, user)
    analysis = db.get(Analysis, analysis_id)
    if not analysis or analysis.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    db.delete(analysis)
    db.commit()
    return MessageOut(detail="Analysis deleted")
