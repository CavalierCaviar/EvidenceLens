"""
Document service: handles PDF upload, parsing, chunking, and embedding storage.
"""
from pathlib import Path
from sqlalchemy.orm import Session
from app.models import Paper, Page, Chunk, Collection
from app.parsing.pdf_parser import parse_pdf
from app.parsing.chunker import chunk_pages
from app.ml.embedder import embed_to_json
from app.config import settings
import shutil
import logging

logger = logging.getLogger(__name__)

def save_and_process_pdf(
    db: Session,
    collection_id: str,
    filename: str,
    pdf_bytes: bytes,
) -> Paper:
    """
    Save PDF, parse it, create Page and Chunk records, embed chunks.
    Returns the Paper record.
    """
    # Parse PDF
    parsed = parse_pdf(pdf_bytes)
    
    # Check for duplicate in this collection (by document_hash)
    existing = db.query(Paper).filter(
        Paper.collection_id == collection_id,
        Paper.document_hash == parsed.document_hash
    ).first()
    if existing:
        return existing  # idempotent
    
    # Save file to disk
    upload_dir = settings.data_path / 'uploads' / collection_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{parsed.document_hash}_{filename}"
    file_path.write_bytes(pdf_bytes)
    
    # Create Paper record
    paper = Paper(
        collection_id=collection_id,
        document_hash=parsed.document_hash,
        filename=filename,
        source_path=str(file_path),
        title=parsed.title,
        authors=parsed.authors,
        year=parsed.year,
        doi=parsed.doi,
        abstract=parsed.abstract,
        study_design=parsed.study_design,
        sample_size=parsed.sample_size,
        metadata_confidence=parsed.metadata_confidence,
        status='parsed',
    )
    db.add(paper)
    db.flush()  # get paper.id
    
    # Create Page records
    for p in parsed.pages:
        page_obj = Page(
            paper_id=paper.id,
            page_number=p.page_number,
            text=p.text,
            section=p.section,
        )
        db.add(page_obj)
    
    # Chunk and embed
    raw_chunks = chunk_pages(parsed.pages)
    for ch in raw_chunks:
        emb_json = ''
        try:
            emb_json = embed_to_json(ch.text)
        except Exception as e:
            logger.warning('Embedding failed for chunk: %s', e)
        
        chunk_obj = Chunk(
            paper_id=paper.id,
            page=ch.page,
            section=ch.section,
            text=ch.text,
            embedding_json=emb_json,
        )
        db.add(chunk_obj)
    
    paper.status = 'processed'
    db.commit()
    db.refresh(paper)
    logger.info('Processed paper %s: %d pages, %d chunks', paper.id, len(parsed.pages), len(raw_chunks))
    return paper


def delete_paper(db: Session, paper: Paper) -> None:
    """Delete paper and its associated file."""
    try:
        if paper.source_path:
            Path(paper.source_path).unlink(missing_ok=True)
    except Exception:
        pass
    db.delete(paper)
    db.commit()


def get_collection_chunks(db: Session, collection_id: str) -> list[dict]:
    """
    Return all chunks in a collection as dicts suitable for retrieval.
    dict keys: chunk_id, paper_id, text, page, section, embedding_json
    """
    papers = db.query(Paper).filter(Paper.collection_id == collection_id).all()
    chunks = []
    for paper in papers:
        for chunk in paper.chunks:
            chunks.append({
                'chunk_id': chunk.id,
                'paper_id': chunk.paper_id,
                'text': chunk.text,
                'page': chunk.page,
                'section': chunk.section,
                'embedding_json': chunk.embedding_json,
            })
    return chunks
