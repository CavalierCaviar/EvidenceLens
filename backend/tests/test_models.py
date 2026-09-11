import pytest
from app.models import User, Collection, Paper, Claim, ClaimPair, EvidenceAssessment, Analysis

def test_create_user(db_session):
    user = User(email="user@test.com", hashed_password="pwd")
    db_session.add(user)
    db_session.commit()
    
    assert user.id is not None
    assert user.email == "user@test.com"

def test_relationships_and_cascade_delete(db_session, test_user):
    collection = Collection(name="Test Col", description="Test", user_id=test_user.id)
    db_session.add(collection)
    db_session.commit()
    
    assert len(test_user.collections) == 1
    
    paper = Paper(
        title="Paper 1", 
        abstract="Abs", 
        collection_id=collection.id,
        document_hash="hash123",
        filename="f.pdf",
        source_path="/path/f.pdf"
    )
    db_session.add(paper)
    db_session.commit()
    
    assert len(collection.papers) == 1
    
    claim = Claim(claim_text="Claim 1", paper_id=paper.id)
    db_session.add(claim)
    db_session.commit()
    
    assert len(paper.claims) == 1
    
    # Cascade delete collection -> papers -> claims
    db_session.delete(collection)
    db_session.commit()
    
    assert db_session.query(Paper).count() == 0
    assert db_session.query(Claim).count() == 0

def test_claim_pair_and_assessment(db_session, test_user):
    collection = Collection(name="Test Col2", user_id=test_user.id)
    db_session.add(collection)
    db_session.commit()
    
    paper1 = Paper(document_hash="h1", filename="1", source_path="1", collection_id=collection.id)
    paper2 = Paper(document_hash="h2", filename="2", source_path="2", collection_id=collection.id)
    db_session.add_all([paper1, paper2])
    db_session.commit()
    
    claim1 = Claim(claim_text="C1", paper_id=paper1.id)
    claim2 = Claim(claim_text="C2", paper_id=paper2.id)
    db_session.add_all([claim1, claim2])
    db_session.commit()
    
    analysis = Analysis(collection_id=collection.id, user_id=test_user.id, research_question="Q1")
    db_session.add(analysis)
    db_session.commit()
    
    pair = ClaimPair(
        analysis_id=analysis.id,
        claim_a_id=claim1.id, 
        claim_b_id=claim2.id, 
        relationship="contradiction"
    )
    db_session.add(pair)
    db_session.commit()
    
    assert pair.id is not None
    
    assessment = EvidenceAssessment(claim_id=claim1.id, evidence_category="Moderate", rubric_score=0.6)
    db_session.add(assessment)
    db_session.commit()
    
    assert assessment.id is not None
