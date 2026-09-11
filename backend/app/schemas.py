from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


RelationshipLabel = Literal["contradiction", "support", "neutral"]
ClaimType = Literal[
    "RESULT",
    "ASSOCIATION",
    "CAUSAL CLAIM",
    "NULL FINDING",
    "METHODOLOGICAL CLAIM",
    "BACKGROUND CLAIM",
]
PipelineMode = Literal[
    "vanilla_rag",
    "rag_claims",
    "rag_contradiction",
    "rag_evidence",
    "full",
]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CollectionCreate(BaseModel):
    name: str
    description: str = ""
    domain_config: str = "biomedical"


class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    domain_config: Optional[str] = None


class CollectionOut(BaseModel):
    id: str
    name: str
    description: str
    domain_config: str
    created_at: datetime
    paper_count: int = 0

    model_config = {"from_attributes": True}


class PaperOut(BaseModel):
    id: str
    collection_id: str
    document_hash: str
    filename: str
    title: str
    authors: str
    year: str
    doi: str
    abstract: str
    study_design: str
    sample_size: str
    metadata_confidence: float
    status: str
    error_message: str
    ingestion_timestamp: datetime

    model_config = {"from_attributes": True}


class PageOut(BaseModel):
    paper_id: str
    page: int
    section: str
    text: str


class ChunkOut(BaseModel):
    id: str
    paper_id: str
    page: int
    section: str
    text: str
    retrieval_score: Optional[float] = None

    model_config = {"from_attributes": True}


class ClaimOut(BaseModel):
    id: str
    paper_id: str
    chunk_id: Optional[str]
    page: int
    source_text: str
    claim_text: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    direction: str
    effect: str
    magnitude: str
    claim_type: str
    statistical_status: str
    effect_size: str
    p_value: str
    confidence_interval: str
    extraction_confidence: float
    paper_title: Optional[str] = None
    evidence_score: Optional[float] = None
    evidence_category: Optional[str] = None

    model_config = {"from_attributes": True}


class ClaimPairOut(BaseModel):
    id: str
    claim_a_id: str
    claim_b_id: str
    semantic_similarity: float
    population_similarity: float
    intervention_similarity: float
    outcome_similarity: float
    nli_entailment_probability: float
    nli_contradiction_probability: float
    nli_neutral_probability: float
    direction_conflict: int
    statistical_conflict: int
    relationship: str
    p_contradiction: float
    p_support: float
    p_neutral: float
    relationship_confidence: float
    contradiction_type: str
    reason: str
    confidence_band: str = "low"
    claim_a: Optional[ClaimOut] = None
    claim_b: Optional[ClaimOut] = None

    model_config = {"from_attributes": True}


class EvidenceExplanationOut(BaseModel):
    assessment_id: str
    claim_id: str
    final_evidence_score: float
    evidence_category: str
    rubric_score: float
    linear_regression_score: float
    study_design: str
    sample_size: str
    precision_score: float
    directness_score: float
    risk_of_bias_score: float
    statistical_support_score: float
    consistency_score: float
    contributing_factors: list[str]
    limitations: list[str]
    note: str = (
        "Estimated evidence strength under the configured framework. "
        "This is not a measure of scientific truth."
    )


class CitationOut(BaseModel):
    statement_id: Optional[str] = None
    paper_id: str
    paper_title: str = ""
    claim_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page: int
    passage: str
    year: str = "UNKNOWN"


class EvidenceItemOut(BaseModel):
    claim: ClaimOut
    paper: PaperOut
    evidence_score: float
    evidence_category: str
    study_design: str
    sample_size: str
    direction: str


class EvidenceSummaryOut(BaseModel):
    overall_strength: float
    consistency: float
    uncertainty: str
    supporting_study_count: int = 0
    contradictory_study_count: int = 0
    neutral_study_count: int = 0
    study_design_distribution: dict[str, int] = Field(default_factory=dict)
    heterogeneity_notes: list[str] = Field(default_factory=list)


class SynthesisOut(BaseModel):
    research_question: str
    overall_finding: str
    supporting_evidence: list[EvidenceItemOut] = Field(default_factory=list)
    contradictory_evidence: list[EvidenceItemOut] = Field(default_factory=list)
    neutral_evidence: list[EvidenceItemOut] = Field(default_factory=list)
    evidence_summary: EvidenceSummaryOut
    contradictions: list[ClaimPairOut] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    sources_of_disagreement: list[str] = Field(default_factory=list)
    uncertainty: str = ""
    grounded: bool = True
    unsupported_claim_rate: float = 0.0
    mode: str = "full"


class QueryRequest(BaseModel):
    research_question: str
    top_k: int = 12
    mode: PipelineMode = "full"


class AnalysisOut(BaseModel):
    id: str
    collection_id: str
    research_question: str
    status: str
    mode: str
    top_k: int
    created_at: datetime
    uncertainty: str
    error_message: str
    synthesis: Optional[SynthesisOut] = None

    model_config = {"from_attributes": True}


class GraphNode(BaseModel):
    id: str
    type: Literal["paper", "claim", "evidence", "question"]
    label: str
    data: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: Literal["supports", "contradicts", "neutral", "derived_from"]
    weight: float = 0.0


class EvidenceGraphOut(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ExperimentCreate(BaseModel):
    name: str
    dataset_version: str = "el-synth-v1"
    mode: PipelineMode = "full"
    retriever: str = "hybrid"
    embedding_model: str = "tfidf-fallback"
    nli_model: str = "heuristic-nli"
    contradiction_classifier: str = "nli_structured_lr"
    evidence_model: str = "linear_regression"
    llm: str = "template"
    top_k: int = 10
    threshold: float = 0.5
    random_seed: int = 42
    evaluation_split: str = "test"


class ExperimentOut(BaseModel):
    id: str
    name: str
    dataset_version: str
    code_version: str
    mode: str
    retriever_configuration: dict[str, Any] = Field(default_factory=dict)
    classifier_configuration: dict[str, Any] = Field(default_factory=dict)
    evidence_model_configuration: dict[str, Any] = Field(default_factory=dict)
    llm_configuration: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    random_seed: int
    evaluation_split: str
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    completed_at: Optional[datetime] = None


class BenchmarkInfo(BaseModel):
    id: str
    version: str
    description: str
    n_questions: int
    n_claim_pairs: int
    n_evidence_items: int


class MessageOut(BaseModel):
    detail: str
