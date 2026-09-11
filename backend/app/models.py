import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("usr"))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    collections: Mapped[list["Collection"]] = relationship(back_populates="owner")


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("col"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    domain_config: Mapped[str] = mapped_column(String(64), default="biomedical")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    owner: Mapped["User"] = relationship(back_populates="collections")
    papers: Mapped[list["Paper"]] = relationship(back_populates="collection", cascade="all, delete-orphan")
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="collection", cascade="all, delete-orphan")


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (UniqueConstraint("collection_id", "document_hash", name="uq_collection_hash"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("P"))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), index=True)
    document_hash: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    source_path: Mapped[str] = mapped_column(String(1024))
    title: Mapped[str] = mapped_column(Text, default="")
    authors: Mapped[str] = mapped_column(Text, default="")
    year: Mapped[str] = mapped_column(String(16), default="UNKNOWN")
    doi: Mapped[str] = mapped_column(String(255), default="UNKNOWN")
    abstract: Mapped[str] = mapped_column(Text, default="")
    study_design: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    sample_size: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    metadata_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    error_message: Mapped[str] = mapped_column(Text, default="")
    ingestion_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    collection: Mapped["Collection"] = relationship(back_populates="papers")
    pages: Mapped[list["Page"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    claims: Mapped[list["Claim"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("pg"))
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    section: Mapped[str] = mapped_column(String(255), default="")

    paper: Mapped["Paper"] = relationship(back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("chk"))
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), index=True)
    page: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(255), default="")
    text: Mapped[str] = mapped_column(Text)
    embedding_json: Mapped[str] = mapped_column(Text, default="")

    paper: Mapped["Paper"] = relationship(back_populates="chunks")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("anl"))
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    research_question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="running")
    mode: Mapped[str] = mapped_column(String(32), default="full")
    top_k: Mapped[int] = mapped_column(Integer, default=12)
    synthesis_json: Mapped[str] = mapped_column(Text, default="{}")
    uncertainty: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    error_message: Mapped[str] = mapped_column(Text, default="")

    collection: Mapped["Collection"] = relationship(back_populates="analyses")
    claims: Mapped[list["Claim"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    pairs: Mapped[list["ClaimPair"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    retrieval_hits: Mapped[list["RetrievalHit"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class RetrievalHit(Base):
    __tablename__ = "retrieval_hits"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("hit"))
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), index=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    dense_score: Mapped[float] = mapped_column(Float, default=0.0)
    lexical_score: Mapped[float] = mapped_column(Float, default=0.0)
    rerank_score: Mapped[float] = mapped_column(Float, default=0.0)

    analysis: Mapped["Analysis"] = relationship(back_populates="retrieval_hits")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("clm"))
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True, nullable=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), nullable=True)
    page: Mapped[int] = mapped_column(Integer, default=1)
    source_text: Mapped[str] = mapped_column(Text, default="")
    claim_text: Mapped[str] = mapped_column(Text)
    population: Mapped[str] = mapped_column(String(255), default="UNKNOWN")
    intervention: Mapped[str] = mapped_column(String(255), default="UNKNOWN")
    comparator: Mapped[str] = mapped_column(String(255), default="UNKNOWN")
    outcome: Mapped[str] = mapped_column(String(255), default="UNKNOWN")
    direction: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    effect: Mapped[str] = mapped_column(String(255), default="UNKNOWN")
    magnitude: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    claim_type: Mapped[str] = mapped_column(String(64), default="RESULT")
    statistical_status: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    effect_size: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    p_value: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    confidence_interval: Mapped[str] = mapped_column(String(128), default="UNKNOWN")
    extraction_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    dataset_overlap: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    author_overlap: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    study_cohort_overlap: Mapped[str] = mapped_column(String(32), default="UNKNOWN")

    paper: Mapped["Paper"] = relationship(back_populates="claims")
    analysis: Mapped["Analysis"] = relationship(back_populates="claims")
    assessment: Mapped["EvidenceAssessment"] = relationship(
        back_populates="claim", uselist=False, cascade="all, delete-orphan"
    )


class ClaimPair(Base):
    __tablename__ = "claim_pairs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("pr"))
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    claim_a_id: Mapped[str] = mapped_column(ForeignKey("claims.id"))
    claim_b_id: Mapped[str] = mapped_column(ForeignKey("claims.id"))
    semantic_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    population_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    intervention_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    outcome_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    nli_entailment_probability: Mapped[float] = mapped_column(Float, default=0.0)
    nli_contradiction_probability: Mapped[float] = mapped_column(Float, default=0.0)
    nli_neutral_probability: Mapped[float] = mapped_column(Float, default=0.0)
    direction_conflict: Mapped[int] = mapped_column(Integer, default=0)
    statistical_conflict: Mapped[int] = mapped_column(Integer, default=0)
    relationship: Mapped[str] = mapped_column(String(32), default="neutral")
    p_contradiction: Mapped[float] = mapped_column(Float, default=0.0)
    p_support: Mapped[float] = mapped_column(Float, default=0.0)
    p_neutral: Mapped[float] = mapped_column(Float, default=0.0)
    relationship_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    contradiction_type: Mapped[str] = mapped_column(String(64), default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    model_variant: Mapped[str] = mapped_column(String(64), default="nli_structured_lr")

    analysis: Mapped["Analysis"] = relationship(back_populates="pairs")


class EvidenceAssessment(Base):
    __tablename__ = "evidence_assessments"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ev"))
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.id"), unique=True)
    study_design: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    sample_size: Mapped[str] = mapped_column(String(32), default="UNKNOWN")
    study_design_score: Mapped[float] = mapped_column(Float, default=0.0)
    log_sample_size: Mapped[float] = mapped_column(Float, default=0.0)
    precision_score: Mapped[float] = mapped_column(Float, default=0.0)
    directness_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_of_bias_score: Mapped[float] = mapped_column(Float, default=0.0)
    statistical_support_score: Mapped[float] = mapped_column(Float, default=0.0)
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)
    expert_score: Mapped[float] = mapped_column(Float, nullable=True)
    rubric_score: Mapped[float] = mapped_column(Float, default=0.0)
    linear_regression_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_evidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_category: Mapped[str] = mapped_column(String(32), default="Weak")
    contributing_factors_json: Mapped[str] = mapped_column(Text, default="[]")
    limitations_json: Mapped[str] = mapped_column(Text, default="[]")
    model_variant: Mapped[str] = mapped_column(String(64), default="linear_regression")

    claim: Mapped["Claim"] = relationship(back_populates="assessment")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("EXP"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    dataset_version: Mapped[str] = mapped_column(String(64), default="el-synth-v1")
    code_version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    mode: Mapped[str] = mapped_column(String(64), default="full")
    retriever_configuration: Mapped[str] = mapped_column(Text, default="{}")
    classifier_configuration: Mapped[str] = mapped_column(Text, default="{}")
    evidence_model_configuration: Mapped[str] = mapped_column(Text, default="{}")
    llm_configuration: Mapped[str] = mapped_column(Text, default="{}")
    hyperparameters: Mapped[str] = mapped_column(Text, default="{}")
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    evaluation_split: Mapped[str] = mapped_column(String(32), default="test")
    status: Mapped[str] = mapped_column(String(32), default="created")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    artifact_paths: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("ann"))
    annotator_id: Mapped[str] = mapped_column(String(64))
    item_id: Mapped[str] = mapped_column(String(64), index=True)
    task_type: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(64), default="")
    score: Mapped[float] = mapped_column(Float, nullable=True)
    adjudicated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
