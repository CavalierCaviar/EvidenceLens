"""Collections CRUD endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Collection, Paper, User
from app.schemas import CollectionCreate, CollectionOut, CollectionUpdate, MessageOut

router = APIRouter()


def _to_out(col: Collection) -> CollectionOut:
    return CollectionOut(
        id=col.id,
        name=col.name,
        description=col.description,
        domain_config=col.domain_config,
        created_at=col.created_at,
        paper_count=len(col.papers),
    )


@router.post("", response_model=CollectionOut, status_code=201)
def create_collection(
    body: CollectionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    col = Collection(
        user_id=user.id,
        name=body.name,
        description=body.description,
        domain_config=body.domain_config,
    )
    db.add(col)
    db.commit()
    db.refresh(col)
    return _to_out(col)


@router.get("", response_model=list[CollectionOut])
def list_collections(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    cols = db.query(Collection).filter(Collection.user_id == user.id).all()
    return [_to_out(c) for c in cols]


@router.get("/{collection_id}", response_model=CollectionOut)
def get_collection(
    collection_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    col = db.get(Collection, collection_id)
    if not col or col.user_id != user.id:
        raise HTTPException(status_code=404, detail="Collection not found")
    return _to_out(col)


@router.patch("/{collection_id}", response_model=CollectionOut)
def update_collection(
    collection_id: str,
    body: CollectionUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    col = db.get(Collection, collection_id)
    if not col or col.user_id != user.id:
        raise HTTPException(status_code=404, detail="Collection not found")
    if body.name is not None:
        col.name = body.name
    if body.description is not None:
        col.description = body.description
    if body.domain_config is not None:
        col.domain_config = body.domain_config
    db.commit()
    db.refresh(col)
    return _to_out(col)


@router.delete("/{collection_id}", response_model=MessageOut)
def delete_collection(
    collection_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    col = db.get(Collection, collection_id)
    if not col or col.user_id != user.id:
        raise HTTPException(status_code=404, detail="Collection not found")
    db.delete(col)
    db.commit()
    return MessageOut(detail="Collection deleted")
