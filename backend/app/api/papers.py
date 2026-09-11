"""Papers endpoints: upload, list, delete, get pages/chunks."""
import asyncio
import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Collection, Paper, User
from app.schemas import MessageOut, PaperOut, PageOut, ChunkOut

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_MB = 50
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def _check_collection(db: Session, collection_id: str, user: User) -> Collection:
    col = db.get(Collection, collection_id)
    if not col or col.user_id != user.id:
        raise HTTPException(status_code=404, detail="Collection not found")
    return col


def _paper_to_out(paper: Paper) -> PaperOut:
    return PaperOut(
        id=paper.id,
        collection_id=paper.collection_id,
        document_hash=paper.document_hash,
        filename=paper.filename,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        doi=paper.doi,
        abstract=paper.abstract,
        study_design=paper.study_design,
        sample_size=paper.sample_size,
        metadata_confidence=paper.metadata_confidence,
        status=paper.status,
        error_message=paper.error_message,
        ingestion_timestamp=paper.ingestion_timestamp,
    )


@router.post("", response_model=list[PaperOut], status_code=201)
async def upload_papers(
    collection_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload one or more PDF files to a collection."""
    _check_collection(db, collection_id, user)

    from app.services.document_service import save_and_process_pdf

    results = []
    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{upload.filename} is not a PDF")
        pdf_bytes = await upload.read()
        if len(pdf_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{upload.filename} exceeds {MAX_UPLOAD_MB}MB limit",
            )
        try:
            paper = save_and_process_pdf(
                db=db,
                collection_id=collection_id,
                filename=upload.filename,
                pdf_bytes=pdf_bytes,
            )
            results.append(_paper_to_out(paper))
        except Exception as e:
            logger.error("Failed to process %s: %s", upload.filename, e)
            raise HTTPException(status_code=500, detail=f"Processing failed for {upload.filename}: {e}")

    return results


@router.get("", response_model=list[PaperOut])
def list_papers(
    collection_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_collection(db, collection_id, user)
    papers = db.query(Paper).filter(Paper.collection_id == collection_id).all()
    return [_paper_to_out(p) for p in papers]


@router.get("/{paper_id}", response_model=PaperOut)
def get_paper(
    collection_id: str,
    paper_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_collection(db, collection_id, user)
    paper = db.get(Paper, paper_id)
    if not paper or paper.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Paper not found")
    return _paper_to_out(paper)


@router.get("/{paper_id}/pages", response_model=list[PageOut])
def get_paper_pages(
    collection_id: str,
    paper_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_collection(db, collection_id, user)
    paper = db.get(Paper, paper_id)
    if not paper or paper.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Paper not found")
    return [
        PageOut(paper_id=p.paper_id, page=p.page_number, section=p.section, text=p.text)
        for p in paper.pages
    ]


@router.delete("/{paper_id}", response_model=MessageOut)
def delete_paper(
    collection_id: str,
    paper_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _check_collection(db, collection_id, user)
    paper = db.get(Paper, paper_id)
    if not paper or paper.collection_id != collection_id:
        raise HTTPException(status_code=404, detail="Paper not found")
    from app.services.document_service import delete_paper as svc_delete
    svc_delete(db, paper)
    return MessageOut(detail="Paper deleted")
