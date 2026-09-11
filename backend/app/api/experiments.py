"""Experiments API: create, list, run, and retrieve experiment results."""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.deps import get_current_user
from app.models import Experiment, User
from app.schemas import (
    BenchmarkInfo, ExperimentCreate, ExperimentOut, MessageOut
)

logger = logging.getLogger(__name__)
router = APIRouter()

SYNTHETIC_BENCHMARK = {
    "id": "el-synth-v1",
    "version": "1.0",
    "description": "Synthetic sanity-test benchmark for EvidenceLens claim classification and evidence estimation.",
    "claim_pairs": [
        {
            "pair_id": "sp001",
            "claim_a": "X increases Y significantly in adults.",
            "claim_b": "X decreases Y significantly in adults.",
            "gold_label": "contradiction",
            "population": "adults",
            "outcome": "Y",
        },
        {
            "pair_id": "sp002",
            "claim_a": "X improves Y in adults over 60.",
            "claim_b": "X improves Y in children under 10.",
            "gold_label": "neutral",
            "population_a": "adults over 60",
            "population_b": "children under 10",
        },
        {
            "pair_id": "sp003",
            "claim_a": "X improves Y.",
            "claim_b": "X significantly improves Y.",
            "gold_label": "support",
        },
        {
            "pair_id": "sp004",
            "claim_a": "X has no significant effect on Y.",
            "claim_b": "X significantly improves Y.",
            "gold_label": "contradiction",
        },
        {
            "pair_id": "sp005",
            "claim_a": "X improves Y.",
            "claim_b": "X improves Z.",
            "gold_label": "neutral",
        },
    ],
    "evidence_items": [
        {
            "item_id": "ev001",
            "claim": "RCT with n=1200 shows X significantly improves Y (p<0.001).",
            "study_design": "Randomized Controlled Trial",
            "sample_size": "1200",
            "p_value": "0.001",
            "expert_score": 0.88,
        },
        {
            "item_id": "ev002",
            "claim": "Pilot study with n=24 shows X may improve Y.",
            "study_design": "Pilot Study",
            "sample_size": "24",
            "p_value": "UNKNOWN",
            "expert_score": 0.28,
        },
        {
            "item_id": "ev003",
            "claim": "Meta-analysis of 15 RCTs shows X significantly improves Y.",
            "study_design": "Meta-analysis",
            "sample_size": "4500",
            "p_value": "0.0001",
            "expert_score": 0.95,
        },
    ],
}


def _experiment_to_out(exp: Experiment) -> ExperimentOut:
    def _parse_json(s: str) -> dict:
        try:
            return json.loads(s) if s else {}
        except Exception:
            return {}

    return ExperimentOut(
        id=exp.id,
        name=exp.name,
        dataset_version=exp.dataset_version,
        code_version=exp.code_version,
        mode=exp.mode,
        retriever_configuration=_parse_json(exp.retriever_configuration),
        classifier_configuration=_parse_json(exp.classifier_configuration),
        evidence_model_configuration=_parse_json(exp.evidence_model_configuration),
        llm_configuration=_parse_json(exp.llm_configuration),
        hyperparameters=_parse_json(exp.hyperparameters),
        random_seed=exp.random_seed,
        evaluation_split=exp.evaluation_split,
        status=exp.status,
        metrics=_parse_json(exp.metrics_json),
        artifact_paths=_parse_json(exp.artifact_paths),
        created_at=exp.created_at,
        completed_at=exp.completed_at,
    )


def _run_experiment_bg(experiment_id: str):
    """Run synthetic benchmark experiment in background."""
    from datetime import datetime, timezone
    from app.ml.contradiction_classifier import classify

    db = SessionLocal()
    try:
        exp = db.get(Experiment, experiment_id)
        if not exp:
            return

        exp.status = "running"
        db.commit()

        benchmark = SYNTHETIC_BENCHMARK
        correct = 0
        total = 0
        contradiction_tp = 0
        contradiction_fp = 0
        contradiction_fn = 0

        for pair in benchmark["claim_pairs"]:
            result = classify(
                claim_a_text=pair["claim_a"],
                claim_b_text=pair["claim_b"],
            )
            predicted = result.relationship
            gold = pair["gold_label"]
            if predicted == gold:
                correct += 1
            if gold == "contradiction" and predicted == "contradiction":
                contradiction_tp += 1
            elif gold != "contradiction" and predicted == "contradiction":
                contradiction_fp += 1
            elif gold == "contradiction" and predicted != "contradiction":
                contradiction_fn += 1
            total += 1

        accuracy = correct / total if total else 0
        precision_c = contradiction_tp / (contradiction_tp + contradiction_fp) if (contradiction_tp + contradiction_fp) > 0 else 0
        recall_c = contradiction_tp / (contradiction_tp + contradiction_fn) if (contradiction_tp + contradiction_fn) > 0 else 0
        f1_c = (2 * precision_c * recall_c) / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0

        # Evidence estimation evaluation
        from app.ml.evidence_engine import assess_evidence, EvidenceFeatures
        import numpy as np

        predicted_scores = []
        expert_scores = []
        for item in benchmark["evidence_items"]:
            ef = EvidenceFeatures(
                study_design=item["study_design"],
                sample_size=item["sample_size"],
                p_value=item["p_value"],
            )
            result_ev = assess_evidence(ef)
            predicted_scores.append(result_ev.final_evidence_score)
            expert_scores.append(item["expert_score"])

        mae = float(np.mean(np.abs(np.array(predicted_scores) - np.array(expert_scores))))

        metrics = {
            "dataset": "el-synth-v1",
            "n_pairs": total,
            "accuracy": round(accuracy, 4),
            "contradiction_precision": round(precision_c, 4),
            "contradiction_recall": round(recall_c, 4),
            "contradiction_f1": round(f1_c, 4),
            "evidence_mae": round(mae, 4),
            "n_evidence_items": len(benchmark["evidence_items"]),
        }

        exp.metrics_json = json.dumps(metrics)
        exp.status = "complete"
        exp.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Experiment %s complete: %s", experiment_id, metrics)

    except Exception as e:
        logger.error("Experiment %s failed: %s", experiment_id, e)
        try:
            exp = db.get(Experiment, experiment_id)
            if exp:
                exp.status = "failed"
                exp.metrics_json = json.dumps({"error": str(e)})
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.get("/benchmark", response_model=BenchmarkInfo)
def get_benchmark_info(user: User = Depends(get_current_user)):
    """Get information about the built-in synthetic benchmark."""
    b = SYNTHETIC_BENCHMARK
    return BenchmarkInfo(
        id=b["id"],
        version=b["version"],
        description=b["description"],
        n_questions=1,
        n_claim_pairs=len(b["claim_pairs"]),
        n_evidence_items=len(b["evidence_items"]),
    )


@router.post("", response_model=ExperimentOut, status_code=201)
def create_experiment(
    body: ExperimentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    exp = Experiment(
        user_id=user.id,
        name=body.name,
        dataset_version=body.dataset_version,
        mode=body.mode,
        retriever_configuration=json.dumps({
            "retriever": body.retriever,
            "embedding_model": body.embedding_model,
            "top_k": body.top_k,
        }),
        classifier_configuration=json.dumps({
            "nli_model": body.nli_model,
            "contradiction_classifier": body.contradiction_classifier,
            "threshold": body.threshold,
        }),
        evidence_model_configuration=json.dumps({
            "evidence_model": body.evidence_model,
        }),
        llm_configuration=json.dumps({"llm": body.llm}),
        hyperparameters=json.dumps({"top_k": body.top_k, "threshold": body.threshold}),
        random_seed=body.random_seed,
        evaluation_split=body.evaluation_split,
        status="created",
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    # Auto-run against synthetic benchmark
    background_tasks.add_task(_run_experiment_bg, exp.id)
    exp.status = "running"
    db.commit()

    return _experiment_to_out(exp)


@router.get("", response_model=list[ExperimentOut])
def list_experiments(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    exps = (
        db.query(Experiment)
        .filter(Experiment.user_id == user.id)
        .order_by(Experiment.created_at.desc())
        .all()
    )
    return [_experiment_to_out(e) for e in exps]


@router.get("/{experiment_id}", response_model=ExperimentOut)
def get_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    exp = db.get(Experiment, experiment_id)
    if not exp or exp.user_id != user.id:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _experiment_to_out(exp)


@router.delete("/{experiment_id}", response_model=MessageOut)
def delete_experiment(
    experiment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    exp = db.get(Experiment, experiment_id)
    if not exp or exp.user_id != user.id:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.delete(exp)
    db.commit()
    return MessageOut(detail="Experiment deleted")
