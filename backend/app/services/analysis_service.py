"""
Analysis service: orchestrates the full EvidenceLens pipeline.
Supports multiple pipeline modes: vanilla_rag, rag_claims, rag_contradiction, rag_evidence, full.
"""
import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Analysis, Claim, ClaimPair, EvidenceAssessment, RetrievalHit, Paper
)
from app.schemas import (
    SynthesisOut, EvidenceSummaryOut, EvidenceItemOut,
    ClaimPairOut, ClaimOut, PaperOut, CitationOut,
)
from app.services.document_service import get_collection_chunks
from app.ml.retriever import retrieve
from app.ml.claim_extractor import extract_claims
from app.ml.contradiction_classifier import classify
from app.ml.evidence_engine import assess_evidence, EvidenceFeatures
from app.ml.synthesizer import synthesize, validate_grounding
from app.ml.embedder import cosine_sim, encode

logger = logging.getLogger(__name__)


def _field_sim(text1: str, text2: str) -> float:
    """Helper to calculate similarity between two text fields."""
    if not text1 or not text2 or text1 == "UNKNOWN" or text2 == "UNKNOWN":
        return 0.5
    try:
        embs = encode([text1, text2])
        return float(cosine_sim(embs[0], embs[1]))
    except Exception:
        return 0.5


def _claim_to_out(c: Claim, paper: Paper | None = None,
                  ea: EvidenceAssessment | None = None) -> ClaimOut:
    """Convert a Claim ORM object to ClaimOut schema."""
    return ClaimOut(
        id=str(c.id),
        paper_id=str(c.paper_id),
        chunk_id=str(c.chunk_id) if c.chunk_id else None,
        page=c.page,
        source_text=c.source_text,
        claim_text=c.claim_text,
        population=c.population,
        intervention=c.intervention,
        comparator=c.comparator,
        outcome=c.outcome,
        direction=c.direction,
        effect=c.effect,
        magnitude=c.magnitude,
        claim_type=c.claim_type,
        statistical_status=c.statistical_status,
        effect_size=c.effect_size,
        p_value=c.p_value,
        confidence_interval=c.confidence_interval,
        extraction_confidence=c.extraction_confidence,
        paper_title=paper.title if paper else None,
        evidence_score=ea.final_evidence_score if ea else None,
        evidence_category=ea.evidence_category if ea else None,
    )


def _paper_to_out(paper: Paper) -> PaperOut:
    """Convert a Paper ORM object to PaperOut schema."""
    return PaperOut(
        id=str(paper.id),
        collection_id=str(paper.collection_id),
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


def _build_evidence_item(
    c: Claim, paper: Paper, ea: EvidenceAssessment | None
) -> EvidenceItemOut:
    """Build an EvidenceItemOut from ORM objects."""
    return EvidenceItemOut(
        claim=_claim_to_out(c, paper, ea),
        paper=_paper_to_out(paper),
        evidence_score=ea.final_evidence_score if ea else 0.0,
        evidence_category=ea.evidence_category if ea else "Very Weak",
        study_design=paper.study_design,
        sample_size=paper.sample_size,
        direction=c.direction,
    )


def _build_evidence_dict(c: Claim, paper: Paper, ea: EvidenceAssessment | None) -> dict:
    """Build a dict for the synthesizer's evidence lists."""
    return {
        "claim_text": c.claim_text,
        "paper_id": str(paper.id),
        "paper_title": paper.title,
        "study_design": paper.study_design,
        "sample_size": paper.sample_size,
        "evidence_score": ea.final_evidence_score if ea else 0.0,
        "direction": c.direction,
    }


def _build_contradiction_dict(
    c1: Claim, c2: Claim, pair: ClaimPair
) -> dict:
    """Build a dict for the synthesizer's contradiction list."""
    return {
        "claim_a_text": c1.claim_text,
        "claim_b_text": c2.claim_text,
        "p_contradiction": pair.p_contradiction,
        "reason": pair.reason,
    }


def _categorize_claim_direction(direction: str) -> str:
    """Categorize a claim's direction into supporting/contradicting/neutral buckets."""
    pos = {"positive", "increase", "improve", "higher", "greater", "beneficial"}
    neg = {"negative", "decrease", "reduce", "lower", "worsened", "null", "no effect"}
    d = direction.lower()
    if d in pos:
        return "supporting"
    elif d in neg:
        return "contradicting"
    return "neutral"


def build_synthesis_out(db: Session, analysis: Analysis) -> SynthesisOut:
    """
    Reconstruct the SynthesisOut schema from the database records.
    """
    claims = db.query(Claim).filter(Claim.analysis_id == analysis.id).all()
    pairs = db.query(ClaimPair).filter(ClaimPair.analysis_id == analysis.id).all()

    supporting_items: list[EvidenceItemOut] = []
    contradicting_items: list[EvidenceItemOut] = []
    neutral_items: list[EvidenceItemOut] = []

    # Dicts for synthesizer
    supporting_dicts: list[dict] = []
    contradicting_dicts: list[dict] = []
    neutral_dicts: list[dict] = []

    citations: list[CitationOut] = []
    study_design_dist: dict[str, int] = {}

    for c in claims:
        paper = db.get(Paper, c.paper_id)
        if paper is None:
            continue
        ea = db.query(EvidenceAssessment).filter(
            EvidenceAssessment.claim_id == c.id
        ).first()

        item = _build_evidence_item(c, paper, ea)
        ev_dict = _build_evidence_dict(c, paper, ea)
        bucket = _categorize_claim_direction(c.direction)

        if bucket == "supporting":
            supporting_items.append(item)
            supporting_dicts.append(ev_dict)
        elif bucket == "contradicting":
            contradicting_items.append(item)
            contradicting_dicts.append(ev_dict)
        else:
            neutral_items.append(item)
            neutral_dicts.append(ev_dict)

        # Track study design distribution
        sd = paper.study_design
        study_design_dist[sd] = study_design_dist.get(sd, 0) + 1

        # Build citation
        citations.append(CitationOut(
            paper_id=str(paper.id),
            paper_title=paper.title,
            claim_id=str(c.id),
            chunk_id=str(c.chunk_id) if c.chunk_id else None,
            page=c.page,
            passage=c.source_text[:500] if c.source_text else "",
            year=paper.year,
        ))

    # Build contradiction pairs output
    contradiction_pairs_out: list[ClaimPairOut] = []
    contradiction_dicts: list[dict] = []
    for p in pairs:
        if p.relationship == "contradiction" and p.p_contradiction >= 0.5:
            c1 = db.get(Claim, p.claim_a_id)
            c2 = db.get(Claim, p.claim_b_id)
            if c1 and c2:
                p1 = db.get(Paper, c1.paper_id)
                p2 = db.get(Paper, c2.paper_id)
                ea1 = db.query(EvidenceAssessment).filter(
                    EvidenceAssessment.claim_id == c1.id).first()
                ea2 = db.query(EvidenceAssessment).filter(
                    EvidenceAssessment.claim_id == c2.id).first()
                contradiction_pairs_out.append(ClaimPairOut(
                    id=str(p.id),
                    claim_a_id=str(p.claim_a_id),
                    claim_b_id=str(p.claim_b_id),
                    semantic_similarity=p.semantic_similarity,
                    population_similarity=p.population_similarity,
                    intervention_similarity=p.intervention_similarity,
                    outcome_similarity=p.outcome_similarity,
                    nli_entailment_probability=p.nli_entailment_probability,
                    nli_contradiction_probability=p.nli_contradiction_probability,
                    nli_neutral_probability=p.nli_neutral_probability,
                    direction_conflict=p.direction_conflict,
                    statistical_conflict=p.statistical_conflict,
                    relationship=p.relationship,
                    p_contradiction=p.p_contradiction,
                    p_support=p.p_support,
                    p_neutral=p.p_neutral,
                    relationship_confidence=p.relationship_confidence,
                    contradiction_type=p.contradiction_type,
                    reason=p.reason,
                    confidence_band=(
                        "high" if p.p_contradiction >= 0.85
                        else "moderate" if p.p_contradiction >= 0.65
                        else "low"
                    ),
                    claim_a=_claim_to_out(c1, p1, ea1),
                    claim_b=_claim_to_out(c2, p2, ea2),
                ))
                contradiction_dicts.append(_build_contradiction_dict(c1, c2, p))

    # Run synthesis via LLM/template
    synth_result = synthesize(
        question=analysis.research_question,
        supporting=supporting_dicts,
        contradicting=contradicting_dicts,
        neutral=neutral_dicts,
        contradictions=contradiction_dicts,
    )

    # Validate grounding
    all_evidence_dicts = supporting_dicts + contradicting_dicts + neutral_dicts
    grounded, unsupported_rate = validate_grounding(synth_result, all_evidence_dicts)

    # Compute evidence summary
    all_scores = []
    for c in claims:
        ea = db.query(EvidenceAssessment).filter(
            EvidenceAssessment.claim_id == c.id).first()
        if ea:
            all_scores.append(ea.final_evidence_score)

    overall_strength = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Consistency: how similar are the scores?
    if len(all_scores) > 1:
        import statistics
        consistency = max(0.0, 1.0 - statistics.stdev(all_scores))
    else:
        consistency = 0.5

    heterogeneity_notes: list[str] = []
    if len(study_design_dist) > 2:
        heterogeneity_notes.append(
            f"Evidence comes from {len(study_design_dist)} different study designs."
        )
    if len(supporting_items) > 0 and len(contradicting_items) > 0:
        heterogeneity_notes.append(
            "Both supporting and contradictory evidence exists."
        )

    evidence_summary = EvidenceSummaryOut(
        overall_strength=overall_strength,
        consistency=consistency,
        uncertainty=synth_result.get("uncertainty", ""),
        supporting_study_count=len(supporting_items),
        contradictory_study_count=len(contradicting_items),
        neutral_study_count=len(neutral_items),
        study_design_distribution=study_design_dist,
        heterogeneity_notes=heterogeneity_notes,
    )

    return SynthesisOut(
        research_question=analysis.research_question,
        overall_finding=synth_result.get("overall_finding", "No finding generated."),
        supporting_evidence=supporting_items,
        contradictory_evidence=contradicting_items,
        neutral_evidence=neutral_items,
        evidence_summary=evidence_summary,
        contradictions=contradiction_pairs_out,
        limitations=synth_result.get("limitations", []),
        citations=citations,
        sources_of_disagreement=synth_result.get("sources_of_disagreement", []),
        uncertainty=synth_result.get("uncertainty", ""),
        grounded=grounded,
        unsupported_claim_rate=unsupported_rate,
        mode=analysis.mode,
    )


def run_analysis(db: Session, analysis: Analysis) -> None:
    """
    Run the full EvidenceLens analysis pipeline.
    """
    try:
        analysis.status = "processing"
        db.commit()

        # a. Get all chunks from the collection
        chunks = get_collection_chunks(db, analysis.collection_id)

        if not chunks:
            analysis.status = "complete"
            analysis.uncertainty = "No documents have been processed in this collection."
            synth = SynthesisOut(
                research_question=analysis.research_question,
                overall_finding="No evidence found. Please upload and process papers first.",
                evidence_summary=EvidenceSummaryOut(
                    overall_strength=0.0, consistency=0.0,
                    uncertainty="No documents available.",
                ),
                mode=analysis.mode,
            )
            analysis.synthesis_json = synth.model_dump_json()
            db.commit()
            return

        # b. Run hybrid retrieval
        hits = retrieve(analysis.research_question, chunks, analysis.top_k)

        # c. Save RetrievalHit records
        retrieval_hits = []
        for hit in hits:
            rh = RetrievalHit(
                analysis_id=analysis.id,
                chunk_id=hit.chunk_id,
                paper_id=hit.paper_id,
                rank=hit.rank,
                dense_score=hit.dense_score,
                lexical_score=hit.lexical_score,
                rerank_score=hit.rerank_score,
            )
            db.add(rh)
            db.flush()
            retrieval_hits.append((rh, hit))

        db_claims: list[Claim] = []

        # d. If mode != 'vanilla_rag': extract claims from top passages
        if analysis.mode != "vanilla_rag":
            for rh, hit in retrieval_hits:
                extracted = extract_claims(hit.text)
                for ext_c in extracted:
                    c = Claim(
                        analysis_id=analysis.id,
                        paper_id=rh.paper_id,
                        chunk_id=rh.chunk_id,
                        page=hit.page,
                        source_text=hit.text[:1000],
                        claim_text=ext_c.claim_text,
                        direction=ext_c.direction,
                        population=ext_c.population,
                        intervention=ext_c.intervention,
                        comparator=ext_c.comparator,
                        outcome=ext_c.outcome,
                        effect=ext_c.effect,
                        magnitude=ext_c.magnitude,
                        claim_type=ext_c.claim_type,
                        statistical_status=ext_c.statistical_status,
                        effect_size=ext_c.effect_size,
                        p_value=ext_c.p_value,
                        confidence_interval=ext_c.confidence_interval,
                        extraction_confidence=ext_c.extraction_confidence,
                    )
                    db.add(c)
                    db.flush()
                    db_claims.append(c)

        # f. If mode includes contradiction: run claim pairing and classification
        if analysis.mode in ("rag_contradiction", "full"):
            for i in range(len(db_claims)):
                for j in range(i + 1, len(db_claims)):
                    c1 = db_claims[i]
                    c2 = db_claims[j]

                    # Only pair claims from DIFFERENT papers
                    if c1.paper_id == c2.paper_id:
                        continue

                    # Semantic similarity gate
                    try:
                        embs = encode([c1.claim_text, c2.claim_text])
                        sim = float(cosine_sim(embs[0], embs[1]))
                    except Exception:
                        sim = 0.5

                    if sim < settings.candidate_similarity_threshold:
                        continue

                    # Run contradiction classifier
                    result = classify(
                        claim_a_text=c1.claim_text,
                        claim_b_text=c2.claim_text,
                        dir_a=c1.direction,
                        dir_b=c2.direction,
                        stat_a=c1.statistical_status,
                        stat_b=c2.statistical_status,
                        pop_a=c1.population,
                        pop_b=c2.population,
                        intervention_a=c1.intervention,
                        intervention_b=c2.intervention,
                        outcome_a=c1.outcome,
                        outcome_b=c2.outcome,
                    )

                    # Compute PICO field similarities
                    pop_sim = _field_sim(c1.population, c2.population)
                    int_sim = _field_sim(c1.intervention, c2.intervention)
                    out_sim = _field_sim(c1.outcome, c2.outcome)

                    cp = ClaimPair(
                        analysis_id=analysis.id,
                        claim_a_id=c1.id,
                        claim_b_id=c2.id,
                        semantic_similarity=sim,
                        population_similarity=pop_sim,
                        intervention_similarity=int_sim,
                        outcome_similarity=out_sim,
                        nli_entailment_probability=result.p_support,
                        nli_contradiction_probability=result.p_contradiction,
                        nli_neutral_probability=result.p_neutral,
                        direction_conflict=1 if result.contradiction_type in ("directional", "direct") else 0,
                        statistical_conflict=1 if result.contradiction_type == "statistical" else 0,
                        relationship=result.relationship,
                        p_contradiction=result.p_contradiction,
                        p_support=result.p_support,
                        p_neutral=result.p_neutral,
                        relationship_confidence=result.relationship_confidence,
                        contradiction_type=result.contradiction_type,
                        reason=result.reason,
                    )
                    db.add(cp)
                    db.flush()

        # h. If mode includes evidence: assess evidence strength for each claim
        if analysis.mode in ("rag_evidence", "full"):
            for c in db_claims:
                paper = db.get(Paper, c.paper_id)
                if paper is None:
                    continue

                # Compute directness scores
                pop_match = _field_sim(c.population, analysis.research_question)
                int_match = _field_sim(c.intervention, analysis.research_question)
                out_match = _field_sim(c.outcome, analysis.research_question)

                feat = EvidenceFeatures(
                    study_design=paper.study_design,
                    sample_size=paper.sample_size,
                    direction=c.direction,
                    statistical_status=c.statistical_status,
                    p_value=c.p_value,
                    confidence_interval=c.confidence_interval,
                    population_match=pop_match,
                    intervention_match=int_match,
                    outcome_match=out_match,
                )

                assessment = assess_evidence(feat)

                ea = EvidenceAssessment(
                    claim_id=c.id,
                    study_design=paper.study_design,
                    sample_size=paper.sample_size,
                    study_design_score=assessment.study_design_score,
                    log_sample_size=assessment.log_sample_size,
                    precision_score=assessment.precision_score,
                    directness_score=assessment.directness_score,
                    risk_of_bias_score=assessment.risk_of_bias_score,
                    statistical_support_score=assessment.statistical_support_score,
                    consistency_score=assessment.consistency_score,
                    rubric_score=assessment.rubric_score,
                    linear_regression_score=assessment.linear_regression_score,
                    final_evidence_score=assessment.final_evidence_score,
                    evidence_category=assessment.evidence_category,
                    contributing_factors_json=json.dumps(assessment.contributing_factors),
                    limitations_json=json.dumps(assessment.limitations),
                )
                db.add(ea)
                db.flush()

        # j. Build full synthesis
        if analysis.mode == "vanilla_rag":
            # For vanilla RAG, just synthesize from raw passages
            passages_as_support = [
                {
                    "claim_text": hit.text,
                    "paper_id": hit.paper_id,
                    "paper_title": hit.paper_id,
                    "study_design": "UNKNOWN",
                    "sample_size": "UNKNOWN",
                    "evidence_score": 0.5,
                    "direction": "unclear",
                }
                for _rh, hit in retrieval_hits
            ]
            synth_result = synthesize(
                question=analysis.research_question,
                supporting=passages_as_support,
                contradicting=[],
                neutral=[],
                contradictions=[],
            )
            grounded, unsupported_rate = validate_grounding(synth_result, passages_as_support)

            synth_out = SynthesisOut(
                research_question=analysis.research_question,
                overall_finding=synth_result.get("overall_finding", ""),
                supporting_evidence=[],
                contradictory_evidence=[],
                neutral_evidence=[],
                evidence_summary=EvidenceSummaryOut(
                    overall_strength=0.0,
                    consistency=0.0,
                    uncertainty=synth_result.get("uncertainty", ""),
                ),
                contradictions=[],
                limitations=synth_result.get("limitations", []),
                citations=[],
                sources_of_disagreement=synth_result.get("sources_of_disagreement", []),
                uncertainty=synth_result.get("uncertainty", ""),
                grounded=grounded,
                unsupported_claim_rate=unsupported_rate,
                mode="vanilla_rag",
            )
        else:
            synth_out = build_synthesis_out(db, analysis)

        # k. Save synthesis and mark complete
        analysis.synthesis_json = synth_out.model_dump_json()
        analysis.uncertainty = synth_out.uncertainty
        analysis.status = "complete"

    except Exception as e:
        logger.exception("Analysis pipeline failed")
        analysis.status = "failed"
        analysis.error_message = str(e)

    # Always commit
    db.commit()
