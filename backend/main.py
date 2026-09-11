"""EvidenceLens — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine
from app.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables on startup
    Base.metadata.create_all(bind=engine)
    # Ensure data directories exist
    _ = settings.data_path
    yield


app = FastAPI(
    title="EvidenceLens API",
    description=(
        "A contradiction-aware, evidence-weighted academic research assistant. "
        "Retrieves, compares, evaluates, and synthesizes findings across scientific literature."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
from app.api import auth, collections, papers, analyses, claims, contradictions, graph, experiments, export  # noqa: E402

app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
app.include_router(collections.router, prefix="/api/collections", tags=["Collections"])
app.include_router(papers.router, prefix="/api/collections/{collection_id}/papers", tags=["Papers"])
app.include_router(analyses.router, prefix="/api/collections/{collection_id}/analyses", tags=["Analyses"])
app.include_router(claims.router, prefix="/api/analyses/{analysis_id}/claims", tags=["Claims"])
app.include_router(contradictions.router, prefix="/api/analyses/{analysis_id}/contradictions", tags=["Contradictions"])
app.include_router(graph.router, prefix="/api/analyses/{analysis_id}/graph", tags=["Graph"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
